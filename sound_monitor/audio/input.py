import functools
import itertools
import logging
import math
import threading
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta
from math import gcd
from queue import Queue

import numpy as np
import sounddevice as sd
from scipy.signal import butter, resample_poly, sosfilt

from sound_monitor.config import Config
from sound_monitor.util.types.singleton import Singleton

_config = Config.get()
_logger = logging.getLogger(__name__)


class Block:
    @staticmethod
    def _get_filter_sos() -> np.ndarray:
        """for bandpass filter"""
        nyquist = _config.uma8_sample_rate / 2
        low, high = _config.audio_bandpass_filter

        return butter(
            4,  # filter order
            [low / nyquist, high / nyquist],
            btype="band",
            output="sos",
        )

    @staticmethod
    def _get_resample_16khz() -> tuple[int, int]:
        """
        for resampling to 16khz

        - yamnet resample rate is 16khz
        - eg up:down = 1:3 for 16k:48k
        """
        source_rate = _config.uma8_sample_rate
        # ---
        g = gcd(16000, source_rate)

        resample_16khz_up = 16000 // g
        resample_16khz_down = source_rate // g

        return resample_16khz_up, resample_16khz_down

    _filter_sos = _get_filter_sos()
    _resample_16khz_up, _resample_16khz_down = _get_resample_16khz()

    def gain(self, data: np.ndarray, gain_factor: float) -> np.ndarray:
        """apply gain to audio data (with clipping to prevent distortion)"""
        return np.clip(data * gain_factor, -1.0, 1.0)

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

        self._gain_data: np.ndarray | None = None
        self._yamnet_data: np.ndarray | None = None
        self._peak_data: np.ndarray | None = None
        self._direction_data: np.ndarray | None = None

    @property
    def gain_data(self) -> np.ndarray:
        if self._gain_data is None:
            self._gain_data = self.gain(self.data, _config.audio_gain_factor)
        return self._gain_data

    @property
    def mono_data(self) -> np.ndarray:
        mono = _config.audio_mono_channel
        return self.gain_data[:, [mono]]

    @property
    def stereo_data(self) -> np.ndarray:
        left, right = _config.audio_stereo_channels
        return self.gain_data[:, [left, right]]

    @property
    def recording_data(self) -> np.ndarray:
        """just extract the stereo channels"""
        return self.stereo_data

    @property
    def yamnet_data(self) -> np.ndarray:
        """mono, resampled to 16khz"""
        if self._yamnet_data is None:
            self._yamnet_data = self.resample_16khz(self.mono_data)
        return self._yamnet_data

    @property
    def peak_data(self) -> np.ndarray:
        """mono, filtered"""
        if self._peak_data is None:
            self._peak_data = self.filter(self.mono_data)
        return self._peak_data

    @property
    def direction_data(self) -> np.ndarray:
        """all channels, filtered"""
        if self._direction_data is None:
            self._direction_data = self.filter(self.gain_data)
        return self._direction_data


class Input(Singleton["Input"]):

    # other things depend on this, so it shouldn't be changed without care
    blocks_per_second: int = 10

    block_seconds: float = 1 / blocks_per_second
    block_size: int = _config.uma8_sample_rate // blocks_per_second
    buffer_size: int = _config.audio_buffer_seconds * blocks_per_second

    def __init__(self) -> None:
        self.error: str | None = None

        self._callbacks: dict[str, dict] = {}
        self._callbacks_lock = threading.RLock()
        "acquisition order: _buffer_lock, _callbacks_lock"

        self._buffer: deque[Block] = deque(maxlen=self.buffer_size)
        self._buffer_lock = threading.RLock()
        "acquisition order: _buffer_lock, _callbacks_lock"

        self._queue: Queue[Block] | None = None
        self._thread: threading.Thread | None = None
        self._stream: sd.InputStream | None = None

    @property
    def time(self) -> float | None:
        with self._buffer_lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[-1].time

    @property
    def clock(self) -> datetime | None:
        with self._buffer_lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[-1].clock

    def start(self) -> None:
        if self._stream is not None:
            return

        with self._callbacks_lock:
            self._callbacks.clear()

        with self._buffer_lock:
            self._buffer.clear()

        self._queue = Queue()

        self._thread = threading.Thread(target=self._worker)
        self._thread.start()

        self._stream = sd.InputStream(
            device=_config.uma8_device_id,
            samplerate=_config.uma8_sample_rate,
            channels=_config.uma8_output_channels,
            dtype=_config.uma8_sample_format,
            callback=self._callback,
            blocksize=self.block_size,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

        if self._queue is not None:
            self._queue.put(None)

        if self._thread not in (None, threading.current_thread()):
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                _logger.error("thread failed to stop")

        with self._callbacks_lock:
            self._callbacks.clear()

        with self._buffer_lock:
            self._buffer.clear()

        self._queue = None
        self._thread = None
        self._stream = None

    def register_callback(
        self,
        name: str,
        callback: Callable[[deque[Block]], None],
        *,
        window: int = 1,  # blocks
        hop: int = 1,  # blocks
        start_time: float | None = None,
    ) -> None:
        """
        register a callback

        args:
        - name: callback name (must be unique)
        - callback: callback function
          - signature: `callback(data: deque[Block])`
        - window: number of blocks to pass to callback
        - hop: number of blocks between calls
        - start_time: start time (stream time)

        notes:
        - window >= hop, hop >= 1
        - start_time should be close to the current time (we don't account for
          potential clock drift or floating point errors)
        """
        if window < hop or hop < 1:
            raise ValueError("need: window >= hop, hop >= 1")

        c = {
            "name": name,
            "callback": callback,
            "window": window,
            "hop": hop,
            "next_call": None,  # call when 0 if not None
            "start_time": start_time,
        }

        if start_time is None:
            c["next_call"] = window

        with self._callbacks_lock:
            self._callbacks[name] = c

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

        self._queue.put(
            Block(
                data=indata.copy(),
                time=time.inputBufferAdcTime,
                clock=now
                - timedelta(seconds=time.currentTime - time.inputBufferAdcTime),
            )
        )

    def _worker(self) -> None:

        def _last_blocks(buffer: deque[Block], stop: int) -> deque[Block]:
            blocks = deque(itertools.islice(reversed(buffer), stop))
            blocks.reverse()
            return blocks

        def _process(
            calls: deque[Callable[[], None]],
            buffer: deque[Block],
            callbacks: dict[str, dict],
        ) -> None:
            for c in callbacks:
                c["next_call"] -= 1
                if c["next_call"] <= 0:
                    calls.append(
                        functools.partial(
                            c["callback"], data=_last_blocks(buffer, c["window"])
                        )
                    )
                    c["next_call"] = c["hop"]

        try:
            while True:
                block = self._queue.get()

                if block is None:
                    self._queue.task_done()
                    return

                calls = deque()

                with self._buffer_lock:
                    with self._callbacks_lock:

                        past_callbacks = deque()

                        for c in self._callbacks.values():
                            if c["start_time"] is None:
                                continue

                            # negative index in the buffer
                            # - equivalent to index relative to `block` which is one past the end of the buffer
                            neg_index = math.floor(
                                (c["start_time"] - block.time) / self.block_seconds
                            )

                            # past
                            if neg_index < 0:

                                # buffer might be empty, this should still work
                                index = len(self._buffer) + neg_index
                                if index < 0:
                                    _logger.warning(
                                        f"{c['name']}: starting {-index * self.block_seconds} seconds late"
                                    )
                                    index = 0

                                c["next_call"] = index + c["window"]

                                past_callbacks.append(c)

                            # present, future
                            else:
                                c["next_call"] = neg_index + c["window"]

                            c["start_time"] = None

                        # process past callbacks
                        if len(past_callbacks) > 0:
                            buffer = deque()
                            for b in self._buffer:
                                buffer.append(b)
                                _process(calls, buffer, past_callbacks)

                        # add to buffer
                        self._buffer.append(block)

                        # process all callbacks
                        _process(calls, self._buffer, self._callbacks)

                self._queue.task_done()

                for c in calls:
                    c()

        except Exception:
            self.error = "error in worker"
            _logger.exception(self.error)
            self.stop()
