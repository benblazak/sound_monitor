import logging

import numpy as np
import sounddevice as sd

from sound_monitor.config import Config
from sound_monitor.util.types.singleton import Singleton

_config = Config.get()
_logger = logging.getLogger(__name__)


class Audio(Singleton["Audio"]):
    def __init__(self) -> None:
        self.block_size: int = _config.uma8_sample_rate // 10

        self.stream: sd.InputStream | None = None

    def init(self) -> None:
        self.start()

    def cleanup(self) -> None:
        self.stop()

    def start(self) -> None:
        if self.stream is not None:
            return

        _logger.debug("starting")

        self.stream = sd.InputStream(
            device=_config.uma8_device_id,
            samplerate=_config.uma8_sample_rate,
            channels=_config.uma8_output_channels,
            dtype=_config.uma8_sample_format,
            callback=self._callback,
            blocksize=self.block_size,
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is None:
            return

        _logger.debug("stopping")

        self.stream.stop()
        self.stream.close()
        self.stream = None

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time,  # CData
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            _logger.warning(f"audio callback status: {status}")

        # TODO
