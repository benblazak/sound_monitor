import logging
import threading
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np
import sounddevice as sd

from sound_monitor.audio.block import Block
from sound_monitor.config import Config
from sound_monitor.util.types.singleton import Singleton

_config = Config.get()
_logger = logging.getLogger(__name__)


class Input(Singleton["Input"]):
    blocks_per_second: int = 10
    block_length: float = 1 / blocks_per_second  # seconds
    block_size: int = _config.uma8_sample_rate // blocks_per_second
    buffer_size: int = _config.audio_buffer_seconds * blocks_per_second

    def __init__(self) -> None:
        self._stream: sd.InputStream | None = None

        self._callbacks: dict[str, dict] = {}
        self._callbacks_lock = threading.Lock()

        self.buffer: deque[Block] = deque(maxlen=self.buffer_size)
        self.buffer_lock = threading.Lock()

    def init(self) -> None:
        self.start()

    def cleanup(self) -> None:
        self.stop()

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
        with self._callbacks_lock:
            self._callbacks[name] = {
                "callback": callback,
                "interval": interval,
                "next_call": interval,  # call when 0
            }

    def remove_callback(self, name: str) -> None:
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

        with self._callbacks_lock:
            for c in self._callbacks.values():
                c["next_call"] -= 1
                if c["next_call"] <= 0:
                    c["callback"](input=self)
                    c["next_call"] = c["interval"]
