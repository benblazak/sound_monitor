import logging
from datetime import datetime
from math import gcd

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from sound_monitor.config import Config

_config = Config.get()
_logger = logging.getLogger(__name__)


class Block:
    @classmethod
    def _init_cls(cls) -> None:
        # bandpass filter
        nyquist = _config.uma8_sample_rate / 2
        low, high = _config.audio_bandpass_filter

        cls._filter_sos = butter(
            4,  # filter order
            [low / nyquist, high / nyquist],
            btype="band",
            output="sos",
        )

        # resample to 16khz
        # - yamnet resample rate is 16khz
        # - eg up:down = 1:3 for 16k:48k
        source_rate = _config.uma8_sample_rate
        # ---
        g = gcd(16000, source_rate)
        cls._resample_16khz_up = 16000 // g
        cls._resample_16khz_down = source_rate // g

    def filter(self, data: np.ndarray) -> np.ndarray:
        """bandpass filter all channels"""
        filtered = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered[:, ch] = sosfilt(self._filter_sos, data[:, ch])
        return filtered

    def resample_16khz(self, data: np.ndarray) -> np.ndarray:
        """resample the data to 16khz"""
        return resample_poly(data, self._resample_16khz_up, self._resample_16khz_down)

    def __init__(self, data: np.ndarray, time: float, clock: datetime) -> None:
        self.data: np.ndarray = data
        """
        audio data -- shape (block_size, channels)
       
        see
        - Stream https://python-sounddevice.readthedocs.io/en/latest/api/streams.html#sounddevice.Stream
          - search `indata`
        """

        self.time: float = time
        """
        capture time -- stream time
        
        see
        - InputStream https://python-sounddevice.readthedocs.io/en/latest/api/streams.html#sounddevice.InputStream
        - Stream https://python-sounddevice.readthedocs.io/en/latest/api/streams.html#sounddevice.Stream
          - search`time.inputBufferAdcTime`
        - time https://python-sounddevice.readthedocs.io/en/latest/api/streams.html#sounddevice.Stream.time
        
        from the docs
        - the current stream time in seconds
        - the time values are monotonically increasing and have unspecified origin
        """

        self.clock: datetime = clock
        """
        capture time -- wall clock time

        calculated from stream times and when the callback was called
        """

    @property
    def mono_data(self) -> np.ndarray:
        mono = _config.audio_mono_channel
        return self.data[:, [mono]]

    @property
    def stereo_data(self) -> np.ndarray:
        left, right = _config.audio_stereo_channels
        return self.data[:, [left, right]]

    @property
    def recording_data(self) -> np.ndarray:
        """just extract the stereo channels"""
        return self.stereo_data

    @property
    def yamnet_data(self) -> np.ndarray:
        """mono, resampled to 16khz"""
        return self.resample_16khz(self.mono_data)

    @property
    def peak_data(self) -> np.ndarray:
        """mono, filtered"""
        return self.filter(self.mono_data)

    @property
    def direction_data(self) -> np.ndarray:
        """all channels, filtered"""
        return self.filter(self.data)


Block._init_cls()
