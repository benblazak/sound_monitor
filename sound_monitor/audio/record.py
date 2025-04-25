import logging
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from typing import Self

from sound_monitor.audio.block import Block
from sound_monitor.audio.input import Input
from sound_monitor.audio.util import audio_trim
from sound_monitor.config import Config

_config = Config.get()
_logger = logging.getLogger(__name__)

_input = Input.get()


class Record:
    def __init__(self, name: str | None = None) -> None:
        self.name: str | None = name
        self.path: Path | None = None

        self.waiting: bool = False
        self.recording: bool = False
        self.want_start_time: float | None = None  # stream time
        self.want_stop_time: float | None = None  # stream time

        self.start_time: float | None = None  # stream time
        self.stop_time: float | None = None  # stream time
        self.start_clock: datetime | None = None  # wall clock time
        self.stop_clock: datetime | None = None  # wall clock time

        self._queue: Queue[Block] = Queue()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._last_block: Block | None = None

    def start(self, time: float | None = None) -> Self:
        if self.recording:
            return

        self.want_start_time = time
        with _input.buffer_lock:
            if time:
                # we have to filter blocks by time in _worker() (since time
                # might be in the future) so enqueue all blocks here
                for block in _input.buffer:
                    self._queue.put(block)

            _input.register_callback(f"record-{id(self)}", self._callback)

        self._thread = threading.Thread(target=self._worker)
        self._thread.start()

        self.waiting = True
        self.recording = True
        return self

    def stop(self, time: float | None = None) -> Self:
        if not self.recording:
            return

        self.want_stop_time = time
        if not time:
            self._queue.put(None)
        self._thread.join()

        self.recording = False
        return self

    def _callback(self, input: Input) -> None:
        self._queue.put(input.buffer[-1])

    def _worker(self) -> None:
        while True:
            block = self._queue.get()
            self._queue.task_done()

            # last block
            if block is None or (
                self.want_stop_time and block.time > self.want_stop_time
            ):
                _input.remove_callback(f"record-{id(self)}")
                while not self._queue.empty():
                    self._queue.get()

                if self._last_block:
                    self.stop_time = self._last_block.time + _input.block_seconds
                    self.stop_clock = self._last_block.clock + timedelta(
                        seconds=_input.block_seconds
                    )

                self._process.stdin.close()
                ret = 0
                try:
                    if self._process:
                        ret = self._process.wait(timeout=5)
                except subprocess.TimeoutExpired as e:
                    _logger.error("ffmpeg failed to stop, sending SIGTERM")
                    self._process.terminate()
                    try:
                        ret = self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired as e:
                        _logger.error("ffmpeg failed to stop, sending SIGKILL")
                        self._process.kill()
                if ret:
                    _logger.error(f"ffmpeg exited with return code {ret}")

                # if late
                if (
                    self._last_block
                    and self.want_stop_time
                    and self._last_block.time > self.want_stop_time
                ):
                    try:
                        audio_trim(
                            self.path, length=self.want_stop_time - self.start_time
                        )
                    except Exception as e:
                        _logger.error(e)

                return

            # first block
            if self.waiting:
                if self.want_start_time:
                    # if early
                    if block.time <= self.want_start_time - _input.block_seconds:
                        continue

                    # if late
                    if block.time > self.want_start_time:
                        _logger.warning(
                            "recording starting late\n"
                            f"  start: {block.time}\n"
                            f"  start wanted: {self.want_start_time}"
                        )

                self.start_time = block.time
                self.start_clock = block.clock
                self.path = _config.data_dir / (
                    _config.prefix(self.start_clock)
                    + (f"_{self.name}" if self.name else "")
                    + ".mp3"
                )
                self._process = subprocess.Popen(
                    _config.record_ffmpeg_command_partial + [self.path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )

                self.waiting = False

            # all blocks
            self._last_block = block
            self._process.stdin.write(block.recording_data.tobytes())
