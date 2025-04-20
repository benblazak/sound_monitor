import logging
import subprocess
from datetime import datetime
from pathlib import Path

from sound_monitor.audio.block import Block
from sound_monitor.audio.input import Input
from sound_monitor.config import Config

_config = Config.get()
_logger = logging.getLogger(__name__)

_input = Input.get()


class Record:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.time: datetime | None = None

        self._process: subprocess.Popen | None = None

    def start(
        self,
        path: Path | None = None,
        time: datetime | None = None,
    ) -> None:
        if self._process is not None:
            return

        self.time = time or datetime.now()
        self.path = path or _config.data_dir / _config.prefix(time) + ".mp3"
        self._process = subprocess.Popen(
            _config.record_ffmpeg_command_partial + [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        for block in _input.buffer:
            if block.time >= self.time:
                self._process.stdin.write(block.recording_data.tobytes())

        _input.register_callback(
            f"record-{id(self)}",
            self._callback,
        )

    def stop(self) -> None:
        if self._process is None:
            return

        _input.remove_callback(f"record-{id(self)}")

        self._process.stdin.close()
        ret = self._process.wait()
        if ret:
            _logger.error(f"ffmpeg exited with code {ret}")
            _logger.error(self._process.stderr.read().decode())
            # TODO might want to email this

    def _callback(self, input: Input) -> None:
        block = input.buffer[-1]
        if block.time >= self.time:
            self._process.stdin.write(block.recording_data.tobytes())
