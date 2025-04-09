import logging
from collections import deque
from datetime import datetime

import numpy as np
import sounddevice as sd
from scipy.signal import butter, resample_poly, sosfilt

from sound_monitor.config import Config
from sound_monitor.util.types.singleton import Singleton

_config = Config.get()
_logger = logging.getLogger(__name__)


class AudioBlock:
    def __init__(self, data: np.ndarray, time: float) -> None:
        self.raw_data: np.ndarray = data  # shape (block_size, channels)
        self.timestamp = datetime.fromtimestamp(time)

        # TODO processed data


class Audio(Singleton["Audio"]):
    def __init__(self) -> None:
        self.blocks_per_second: int = 10
        self.block_size: int = _config.uma8_sample_rate // self.blocks_per_second
        self.buffer_size: int = _config.audio_buffer_seconds * self.blocks_per_second
        self.buffer: deque = deque(maxlen=self.buffer_size)

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

        self.buffer.append(AudioBlock(indata.copy(), time.inputBufferAdcTime))
