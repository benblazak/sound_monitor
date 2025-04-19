import logging
from collections import deque
from datetime import datetime
from math import gcd

import numpy as np
import sounddevice as sd
from scipy.signal import butter, resample_poly, sosfilt

from sound_monitor.config import Config
from sound_monitor.util.types.singleton import Singleton

_config = Config.get()
_logger = logging.getLogger(__name__)


class AudioBlock:
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
        yamnet_rate = 16000  # yamnet requires 16khz input
        source_rate = _config.uma8_sample_rate
        g = gcd(yamnet_rate, source_rate)

        cls._resample_16khz_up = yamnet_rate // g
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

    def __init__(self, data: np.ndarray, time: float) -> None:
        self.data: np.ndarray = data  # shape (block_size, channels)
        self.timestamp = datetime.fromtimestamp(time)

        self._yamnet_data: np.ndarray | None = None
        self._direction_data: np.ndarray | None = None
        self._recording_data: np.ndarray | None = None

    @property
    def mono_data(self) -> np.ndarray:
        mono = _config.audio_mono_channel
        return self.data[:, [mono]]

    @property
    def stereo_data(self) -> np.ndarray:
        left, right = _config.audio_stereo_channels
        return self.data[:, [left, right]]

    @property
    def yamnet_data(self) -> np.ndarray:
        """mono, filtered, resampled to 16khz"""
        if self._yamnet_data is None:
            self._yamnet_data = self.resample_16khz(self.filter(self.mono_data))
        return self._yamnet_data

    @property
    def direction_data(self) -> np.ndarray:
        """all channels, filtered"""
        if self._direction_data is None:
            self._direction_data = self.filter(self.data)
        return self._direction_data

    @property
    def recording_data(self) -> np.ndarray:
        """just extract the stereo channels"""
        if self._recording_data is None:
            self._recording_data = self.stereo_data
        return self._recording_data


AudioBlock._init_cls()


class Input(Singleton["Input"]):
    block_size: int = _config.uma8_sample_rate // _config.audio_blocks_per_second
    buffer_size: int = _config.audio_buffer_seconds * _config.audio_blocks_per_second

    def __init__(self) -> None:
        self.buffer: deque[AudioBlock] = deque(maxlen=self.buffer_size)
        self.stream: sd.InputStream | None = None
        self.callbacks: dict[str, dict] = {}

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

    def register_callback(
        self,
        name: str,
        callback: callable,
        interval: int,  # blocks
    ) -> None:
        """
        register a callback

        args
        - name: must be unique
        - callback:
          - args: this Input object
          - returns: None
        - interval: number of new blocks availble before next call
          - see config for number of blocks per second
        """
        self.callbacks[name] = {
            "callback": callback,
            "interval": interval,
            "next_call": interval,  # call when 0
        }

    def remove_callback(self, name: str) -> None:
        if name in self.callbacks:
            del self.callbacks[name]

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

        for c in self.callbacks.values():
            c["next_call"] -= 1
            if c["next_call"] <= 0:
                c["callback"](self)
                c["next_call"] = c["interval"]
