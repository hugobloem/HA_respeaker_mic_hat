#!/usr/bin/env python3
"""Runs an Assist pipeline in a loop, executing voice commands and printing TTS response URLs."""
from __future__ import annotations

import argparse
import asyncio
import audioop
import logging
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
import pvporcupine
import pyaudio

from pixels import Pixels

_LOGGER = logging.getLogger(__name__)


@dataclass
class State:
    """Client state."""

    args: argparse.Namespace
    running: bool = True
    recording: bool = False
    hotword_detected: bool = False
    pipeline_running: bool = False
    speaking: bool = False
    audio_queue: asyncio.Queue[bytes] = field(default_factory=asyncio.Queue)
    tts_url: str = ""


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rate",
        default=16000,
        type=int,
        help="Rate of input audio (hertz)",
    )
    parser.add_argument(
        "--width",
        default=2,
        type=int,
        help="Width of input audio samples (bytes)",
    )
    parser.add_argument(
        "--channels",
        default=1,
        type=int,
        help="Number of input audio channels",
    )
    parser.add_argument(
        "--device_index",
        default=1,
        type=int,
        help="Device index",
    )
    parser.add_argument(
        "--speaker_index",
        default=1,
        type=int,
        help="Speaker index",
    )
    parser.add_argument(
        "--samples-per-chunk",
        type=int,
        default=1024,
        help="Number of samples to read at a time from stdin",
    )
    #
    parser.add_argument("--token", required=True, help="HA auth token")
    parser.add_argument(
        "--pipeline", help="Name of HA pipeline to use (default: preferred)"
    )
    parser.add_argument(
        "--server", default="localhost:8123", help="host:port of HA server"
    )
    #
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print DEBUG messages to console",
    )
    parser.add_argument(
        "--hotword",
        default="computer",
        help="Hot word to listen for",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    # Start reading raw audio from stdin
    state = State(args=args)
    audio_thread = threading.Thread(
        target=read_audio,
        args=(
            state,
            asyncio.get_running_loop(),
        ),
        daemon=True,
    )
    lights_thread = threading.Thread(
        target=lights_control,
        args=(
            state,
            asyncio.get_running_loop(),
        ),
        daemon=True,
    )
    audio_thread.start()
    lights_thread.start()
    try:
        while True:
            await loop_pipeline(state)
    except KeyboardInterrupt:
        pixels = Pixels()
        pixels.off()
        pass
    finally:
        state.recording = False
        state.running = False
        state.hotword_detected = False
        audio_thread.join()
        lights_thread.join()


async def loop_pipeline(state: State) -> None:
    """Run pipeline in a loop, executing voice commands and printing TTS URLs."""
    args = state.args

    hotword = Hotword(state)
    # state.hotword_detected = False
    # state.pipeline_running = False
    # state.speaking = False

    url = f"ws://{args.server}/api/websocket"
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as websocket:
            _LOGGER.debug("Authenticating")
            msg = await websocket.receive_json()
            assert msg["type"] == "auth_required", msg

            await websocket.send_json(
                {
                    "type": "auth",
                    "access_token": args.token,
                }
            )

            msg = await websocket.receive_json()
            _LOGGER.debug(msg)
            assert msg["type"] == "auth_ok", msg
            _LOGGER.info("Authenticated")

            message_id = 1
            pipeline_id: Optional[str] = None
            if args.pipeline:
                # Get list of available pipelines and resolve name
                await websocket.send_json(
                    {
                        "type": "assist_pipeline/pipeline/list",
                        "id": message_id,
                    }
                )
                msg = await websocket.receive_json()
                _LOGGER.debug(msg)
                message_id += 1

                pipelines = msg["result"]["pipelines"]
                for pipeline in pipelines:
                    if pipeline["name"] == args.pipeline:
                        pipeline_id = pipeline["id"]
                        break

                if not pipeline_id:
                    raise ValueError(
                        f"No pipeline named {args.pipeline} in {pipelines}"
                    )

            # Hotword loop
            state.hotword_detected = False
            while not state.hotword_detected:
                state.recording = True

                # Clear audio queue
                while not state.audio_queue.empty():
                    state.audio_queue.get_nowait()

                while not state.hotword_detected:
                    audio_chunk = await state.audio_queue.get()
                    hotword.detect(audio_chunk)

            # Pipeline loop
            state.hotword_detected = False
            state.pipeline_running = True
            while state.pipeline_running:

                # Run pipeline
                _LOGGER.debug("Starting pipeline")
                pipeline_args = {
                    "type": "assist_pipeline/run",
                    "id": message_id,
                    "start_stage": "stt",
                    "end_stage": "tts",
                    "input": {
                        "sample_rate": 16000,
                    },
                }
                if pipeline_id:
                    pipeline_args["pipeline"] = pipeline_id
                await websocket.send_json(pipeline_args)
                message_id += 1

                msg = await websocket.receive_json()
                _LOGGER.debug(msg)
                assert msg["success"], "Pipeline failed to run"

                # Get handler id.
                # This is a single byte prefix that needs to be in every binary payload.
                msg = await websocket.receive_json()
                _LOGGER.debug(msg)
                handler_id = bytes(
                    [msg["event"]["data"]["runner_data"]["stt_binary_handler_id"]]
                )

                # Audio loop for single pipeline run
                receive_event_task = asyncio.create_task(websocket.receive_json())
                while True:
                    audio_chunk = await state.audio_queue.get()

                    # Prefix binary message with handler id
                    send_audio_task = asyncio.create_task(
                        websocket.send_bytes(handler_id + audio_chunk)
                    )
                    pending = {send_audio_task, receive_event_task}
                    done, pending = await asyncio.wait(
                        pending,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if receive_event_task in done:
                        event = receive_event_task.result()
                        _LOGGER.debug(event)
                        event_type = event["event"]["type"]
                        if event_type == "run-end":
                            state.recording = False
                            state.pipeline_running = False
                            _LOGGER.debug("Pipeline finished")
                            break

                        if event_type == "error":
                            state.pipeline_running = False
                            raise RuntimeError(event["event"]["data"]["message"])

                        if event_type == "tts-end":
                            # URL of text to speech audio response (relative to server)
                            tts_url = event["event"]["data"]["tts_output"]["url"]

                            _LOGGER.debug(tts_url)
                            play_audio(state, tts_url)
                            

                        receive_event_task = asyncio.create_task(
                            websocket.receive_json()
                        )

                    if send_audio_task not in done:
                        await send_audio_task


def read_audio(state: State, loop: asyncio.AbstractEventLoop) -> None:
    """Reads chunks of raw audio from standard input."""
    try:
        args = state.args
        bytes_per_chunk = args.samples_per_chunk * args.width * args.channels
        samples_per_chunk = args.samples_per_chunk
        rate = args.rate
        width = args.width
        channels = args.channels
        device_index = args.device_index
        ratecv_state = None

        p = pyaudio.PyAudio()

        stream = p.open(
                    rate=rate,
                    format=p.get_format_from_width(width),
                    channels=channels,
                    input=True,
                    input_device_index=device_index,)


        _LOGGER.debug("Reading audio from pyaudio")

        while True:
            chunk = stream.read(samples_per_chunk, exception_on_overflow=False)
            if (not chunk) or (not state.running):
                # Signal other thread to stop
                state.audio_queue.put_nowait(bytes())
                break

            if state.recording:
                # Convert to 16Khz, 16-bit, mono
                if channels != 1:
                    chunk = audioop.tomono(chunk, width, 1.0, 1.0)

                if width != 2:
                    chunk = audioop.lin2lin(chunk, width, 2)

                if rate != 16000:
                    chunk, ratecv_state = audioop.ratecv(
                        chunk,
                        2,
                        1,
                        rate,
                        16000,
                        ratecv_state,
                    )

                # Pass converted audio to loop
                loop.call_soon_threadsafe(state.audio_queue.put_nowait, chunk)
    except Exception:
        _LOGGER.exception("Unexpected error reading audio")

class Hotword:
    def __init__(self, state: State) -> None:
        self.state = state
        self.args = state.args
        self.keyword = self.args.hotword

        self.porcupine = pvporcupine.create(
            keywords=[self.keyword],
        )
        self.chunk_size = self.porcupine.frame_length * 2
        self.chunk_format = "h" * self.porcupine.frame_length


    def detect(self, audio_buffer) -> None:
        """Detect if the hotword has been called"""
        try:

            while  len(audio_buffer) >= self.chunk_size:
                chunk = audio_buffer[: self.chunk_size]
                audio_buffer = audio_buffer[
                    self.chunk_size :
                ]

                unpacked_chunk = struct.unpack_from(self.chunk_format, chunk)
                keyword_index = self.porcupine.process(unpacked_chunk)
                
                if keyword_index >= 0:
                    _LOGGER.debug("Hotword detected")
                    self.state.hotword_detected = True

        except Exception:
            _LOGGER.exception("Unexpected error detecting hotword")


def play_audio(state: State, tts_url) -> None:
    """Plays wav audio from the tts url"""
    try:
        state.speaking = True
        args = state.args
        server = args.server

        _LOGGER.debug("Playing audio from url")

        url = "http://"+server+tts_url
        if tts_url.endswith('.mp3'):
            cmd = ["mpg123", "-q", url]
        elif tts_url.endswith('.wav'):
            cmd = ["curl", "url", "|", "aplay"]
        else:
            raise ValueError(f"Unknown audio format: {tts_url}")
        
        subprocess.call(cmd)
        state.speaking = False

    except Exception:
        _LOGGER.exception("Unexpected error playing audio")
    

def lights_control(state: State, loop: asyncio.AbstractEventLoop) -> None:
    """Control the lights based on the state"""
    try:
        pixels = Pixels()

        status = ""
        while True:
            if state.hotword_detected:
                if status != "hotword":
                    _LOGGER.debug("Lights: think")
                    pixels.think()
                    status = "hotword"
            elif state.pipeline_running:
                if status != "listening":
                    _LOGGER.debug("Lights: listen")
                    pixels.listen()
                    status = "listening"
            elif state.speaking:
                if status != "speaking":
                    _LOGGER.debug("Lights: speak")
                    pixels.speak()
                    status = "speaking"
            else:
                if status != "off":
                    _LOGGER.debug("Lights: off")
                    pixels.off()
                    status = "off"

            time.sleep(0.1)
    except Exception:
        _LOGGER.exception("Unexpected error controlling lights")

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
