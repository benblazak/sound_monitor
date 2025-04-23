import logging
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from typing import Self

from sound_monitor.audio.block import Block
from sound_monitor.audio.input import Input
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

        if time:
            self.want_stop_time = time
        else:
            self._queue.put(None)
        self._thread.join()
        self._process.stdin.close()
        ret = self._process.wait()
        if ret:
            _logger.error(
                f"ffmpeg exited with code {ret}\n"
                + self._process.stderr.read().decode().indent(2)
            )
            raise RuntimeError(f"ffmpeg exited with code {ret}")

        if time is not None:
            length = self.last_block_time + _input.block_seconds - self.first_block_time
            trim_length = time - self.first_block_time
            if trim_length >= length:
                _logger.warning(
                    "cannot trim\n"
                    f"  length: {length}\n"
                    f"  length wanted: {trim_length}"
                )
                return
            else:
                trim_path = self.path.with_suffix(".tmp.mp3")
                trim_process = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-i",
                        str(self.path),
                        "-t",
                        str(trim_length.total_seconds()),
                        "-c",
                        "copy",  # stream copy to avoid re-encoding
                        str(trim_path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                ret = trim_process.wait()
                if ret:
                    trim_path.unlink()
                    _logger.error(
                        f"ffmpeg trim exited with code {ret}\n"
                        + trim_process.stderr.read().decode().indent(2)
                    )

                trim_path.rename(self.path)

        self.recording = False
        return self

    def _callback(self, input: Input) -> None:
        self._queue.put(input.buffer[-1])

    def _worker(self) -> None:
        while True:
            block = self._queue.get()

            # signal to end
            if block is None or (
                self.want_stop_time and self.want_stop_time < block.time
            ):
                _input.remove_callback(f"record-{id(self)}")
                while not self._queue.empty():
                    self._queue.get()

                self.stop_time = self._last_block.time + _input.block_seconds
                self.stop_clock = self._last_block.clock + timedelta(
                    seconds=_input.block_seconds
                )
                return

            # first block
            if self.waiting:
                if self.want_start_time:
                    # if early
                    if block.time <= (self.want_start_time - _input.block_seconds):
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
