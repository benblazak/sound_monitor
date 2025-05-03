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

        self.start_time: float | None = None  # stream time
        self.stop_time: float | None = None  # stream time
        self.start_clock: datetime | None = None  # wall clock time
        self.stop_clock: datetime | None = None  # wall clock time

        self._queue: Queue[Block] | None = None
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._last_block: Block | None = None

    def start(self, time: float | None = None) -> Self:
        if self._queue is not None:
            raise RuntimeError("already started")

        self._queue = Queue()
        with _input.buffer_lock:
            if not _input.buffer:
                self.stop()
                raise RuntimeError("input buffer is empty")

            first_block = None
            if time:
                for block in _input.buffer:
                    if block.time > time - _input.block_seconds:
                        self._queue.put(block)
                        if first_block is None:
                            first_block = block
            # if no time
            # if no block found (time is in the future)
            if first_block is None:
                first_block = _input.buffer[-1]
                self._queue.put(first_block)

            _input.register_callback(f"record-{id(self)}", self._callback)

        if time:
            if first_block.time + _input.block_seconds < time:
                _logger.warning(
                    "starting early\n"
                    f"  wanted: {time}\n"
                    f"  starting: {first_block.time}"
                )
            if first_block.time > time:
                _logger.warning(
                    "starting late\n"
                    f"  wanted: {time}\n"
                    f"  starting: {first_block.time}"
                )

        self.start_time = first_block.time
        self.start_clock = first_block.clock

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
        if self._process.poll() is not None:
            self.stop()
            raise RuntimeError("ffmpeg failed to start")

        self._thread = threading.Thread(target=self._worker)
        self._thread.start()

        return self

    def stop(self, time: float | None = None) -> Self:
        _input.remove_callback(f"record-{id(self)}")

        if self._queue:
            self._queue.put(None)

        if self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                _logger.error("thread failed to stop")

        if self._process:
            self._process.stdin.close()
            try:
                ret = self._process.wait(timeout=5)
                if ret:
                    stderr = self._process.stderr.read().decode().strip()
                    _logger.error(
                        f"ffmpeg exited with return code {ret}\n" + stderr.indent(2)
                    )
            except subprocess.TimeoutExpired:
                _logger.error("ffmpeg failed to stop, sending SIGTERM")
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _logger.error("ffmpeg failed to stop, sending SIGKILL")
                    self._process.kill()

        if self._last_block:
            self.stop_time = self._last_block.time + _input.block_seconds
            self.stop_clock = self._last_block.clock + timedelta(
                seconds=_input.block_seconds
            )

        if not self.path.exists():
            _logger.error(f"no recording at {self.path}")

        if time:
            try:
                if time < self.start_time:
                    _logger.warning(
                        "trim: deleting file\n"
                        f"  wanted stop: {time}\n"
                        f"  start: {self.start_time}\n"
                        f"  stop: {self.stop_time}"
                    )
                    self.path.unlink(missing_ok=True)
                elif time > self.stop_time:
                    _logger.warning(
                        "trim: doing nothing\n"
                        f"  start: {self.start_time}\n"
                        f"  stop: {self.stop_time}\n"
                        f"  wanted stop: {time}"
                    )
                else:
                    trim_length = time - self.start_time
                    audio_trim(self.path, length=trim_length)
                    self.stop_time = time
                    self.stop_clock = self.stop_clock - timedelta(
                        seconds=self.stop_time - time
                    )
            except Exception as e:
                _logger.error(f"trim failed: {e}")

        self._queue = None
        self._thread = None
        self._process = None
        self._last_block = None

        return self

    def _callback(self, input: Input) -> None:
        self._queue.put(input.buffer[-1])

    def _worker(self) -> None:
        try:
            while True:
                block = self._queue.get()
                self._queue.task_done()

                if block is None:
                    return

                self._last_block = block
                self._process.stdin.write(block.recording_data.tobytes())

        except Exception as e:
            _logger.error(f"error in worker: {e}")
            self.stop()
