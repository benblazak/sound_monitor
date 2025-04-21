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
        self.time: datetime | None = None
        self.first_block_time: datetime | None = None
        self.last_block_time: datetime | None = None

        self._queue: Queue[Block] = Queue()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None

    def start(self, time: datetime | None = None) -> Self:
        if self._process is not None:
            raise RuntimeError("already recording")

        self.time = time or datetime.now()

        with _input.buffer_lock:
            for block in _input.buffer:
                self._queue.put(block)

            _input.register_callback(
                f"record-{id(self)}",
                self._callback,
            )

        self._thread = threading.Thread(target=self._worker)
        self._thread.start()

        return self

    def stop(self, time: datetime | None = None) -> None:
        if self._process is None:
            return

        _input.remove_callback(f"record-{id(self)}")
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
            # TODO get length from ffprobe in util.audio_length?
            length = self.last_block_time + _input.block_length - self.first_block_time
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

    def _callback(self, input: Input) -> None:
        self._queue.put(input.buffer[-1])

    def _worker(self) -> None:
        while True:
            block = self._queue.get()

            # signal to end
            if block is None:
                return

            # first block
            if self.first_block_time is None:
                # if early
                if block.time < self.time:
                    continue

                # if late
                if block.time - self.time > _input.block_length * 1.5:
                    _logger.warning(
                        "recording starting late\n"
                        f"  start: {block.time}\n"
                        f"  start wanted: {self.time}"
                    )

                self.first_block_time = block.time
                self.path = _config.data_dir / (
                    _config.prefix(self.first_block_time)
                    + (f"_{self.name}" if self.name else "")
                    + ".mp3"
                )
                self._process = subprocess.Popen(
                    _config.record_ffmpeg_command_partial + [self.path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )

            # all blocks
            self.last_block_time = block.time
            self._process.stdin.write(block.recording_data.tobytes())
