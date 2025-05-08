import itertools
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta
from math import gcd

import numpy as np
import sounddevice as sd
from scipy.signal import butter, resample_poly, sosfilt

from sound_monitor.config import Config
from sound_monitor.util.types.singleton import Singleton

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


class Input(Singleton["Input"]):

    # other things depend on this, so it shouldn't be changed without care
    blocks_per_second: int = 10

    block_seconds: float = 1 / blocks_per_second
    block_size: int = _config.uma8_sample_rate // blocks_per_second
    buffer_size: int = _config.audio_buffer_seconds * blocks_per_second

    def __init__(self) -> None:
        self._stream: sd.InputStream | None = None

        self._callbacks: dict[str, dict] = {}
        self._callbacks_lock = threading.Lock()

        # lock for writes (in this file), or to (very quickly) pause writes
        self.buffer: deque[Block] = deque(maxlen=self.buffer_size)
        self.buffer_lock = threading.Lock()

    @property
    def time(self) -> float:
        return self.buffer[-1].time

    @property
    def clock(self) -> datetime:
        return self.buffer[-1].clock

    @property
    def last_block(self) -> Block:
        return self.buffer[-1]

    def last_blocks(self, stop: int) -> deque[Block]:
        blocks = deque(itertools.islice(reversed(self.buffer), stop))
        blocks.reverse()
        return blocks

    def start(self) -> None:
        if self._stream is not None:
            return

        _logger.debug("starting")

        self._stream = sd.InputStream(
            device=_config.uma8_device_id,
            samplerate=_config.uma8_sample_rate,
            channels=_config.uma8_output_channels,
            dtype=_config.uma8_sample_format,
            callback=self._callback,
            blocksize=self.block_size,
        )
        self._stream.start()

        while not len(self.buffer):
            time.sleep(self.block_seconds / 2)

    def stop(self) -> None:
        if self._stream is None:
            return

        _logger.debug("stopping")

        self._stream.stop()
        self._stream.close()
        self._stream = None

    def register_callback(
        self,
        name: str,
        callback: Callable[["Input"], None],
        *,
        interval: int = 1,  # blocks
    ) -> None:
        """
        register a callback

        notes:
        - the buffer won't be modified while the callbacks are running, so don't
          hold the buffer lock (or modify the buffer) (just read from it)
        """
        with self._callbacks_lock:
            self._callbacks[name] = {
                "callback": callback,
                "interval": interval,
                "next_call": interval,  # call when 0
            }

    def remove_callback(self, name: str) -> None:
        """
        remove a callback
        """
        with self._callbacks_lock:
            if name in self._callbacks:
                del self._callbacks[name]

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time,  # CData
        status: sd.CallbackFlags,
    ) -> None:

        now = datetime.now()

        if status:
            _logger.warning(f"audio callback status: {status}")

        with self.buffer_lock:
            self.buffer.append(
                Block(
                    data=indata.copy(),
                    time=time.inputBufferAdcTime,
                    clock=now
                    - timedelta(seconds=time.currentTime - time.inputBufferAdcTime),
                )
            )

        callbacks = []
        with self._callbacks_lock:
            for c in self._callbacks.values():
                c["next_call"] -= 1
                if c["next_call"] <= 0:
                    callbacks.append(c)
                    c["next_call"] = c["interval"]
        for c in callbacks:
            c["callback"](input=self)
