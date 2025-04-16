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
    # yamnet resample rate
    # - eg up:down = 1:3 for 16k:48k
    _resample_yamnet_rate = 16000  # yamnet requires 16khz input
    _resample_source_rate = _config.uma8_sample_rate
    _resample_gcd = gcd(_resample_yamnet_rate, _resample_source_rate)
    _resample_up = _resample_yamnet_rate // _resample_gcd
    _resample_down = _resample_source_rate // _resample_gcd

    # bandpass filter
    _bandpass_nyquist = _config.uma8_sample_rate / 2
    _bandpass_low, _bandpass_high = _config.audio_bandpass_filter
    _bandpass_sos = butter(
        4,  # filter order
        [_bandpass_low / _bandpass_nyquist, _bandpass_high / _bandpass_nyquist],
        btype="band",
        output="sos",
    )

    def __init__(self, data: np.ndarray, time: float) -> None:
        self.raw_data: np.ndarray = data  # shape (block_size, channels)
        self.timestamp = datetime.fromtimestamp(time)

        # yamnet data: mono, filtered, resampled to 16khz
        mono_channel = _config.audio_mono_channel
        mono_data = data[:, mono_channel]
        mono_filtered = sosfilt(self._bandpass_sos, mono_data)
        self.yamnet_data = resample_poly(
            mono_filtered, self._resample_up, self._resample_down
        )

        # direction data: all 7 channels, filtered
        self.direction_data = np.zeros_like(data)
        for ch in range(data.shape[1]):
            self.direction_data[:, ch] = sosfilt(self._bandpass_sos, data[:, ch])

        # recording data: just extract the stereo channels
        left, right = _config.audio_stereo_channels
        self.recording_data = np.vstack((data[:, left], data[:, right])).T


class Input(Singleton["Input"]):
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
