# TODO from claude, review

import functools
import logging
import math
import threading
from collections import deque
from collections.abc import Callable
from datetime import datetime
from queue import Queue

import numpy as np

from sound_monitor.audio.input import Block, Input
from sound_monitor.config import Config
from sound_monitor.util.types.lifecycle import Lifecycle, State
from sound_monitor.util.types.singleton import Singleton

_config = Config.get()
_logger = logging.getLogger(__name__)

_input = Input.get()


class Levels:
    """
    RMS levels and decibel calculations for a time period.
    """

    def __init__(
        self,
        rms: float,
        db: float,
        time: float,
        utc: datetime,
    ) -> None:
        self.rms: float = rms
        """root mean square amplitude (0.0 to 1.0)"""

        self.db: float = db
        """decibel level (dB SPL, relative to reference)"""

        self.time: float = time
        """stream time of measurement"""

        self.utc: datetime = utc
        """utc time of measurement"""

    @property
    def clock(self) -> datetime:
        """local time of measurement"""
        return self.utc.astimezone(_config.timezone)


class RMS(Singleton["RMS"]):
    """
    Real-time RMS and decibel level monitoring for background noise tracking.

    Calculates continuous background noise levels at regular intervals,
    separate from peak detection which focuses on transient events.
    """

    # analysis parameters
    window_seconds: float = 1.0  # 1 second analysis windows
    hop_seconds: float = 1.0  # update every 1 second

    # convert to blocks
    window_blocks = math.ceil(window_seconds * _input.blocks_per_second)
    hop_blocks = round(hop_seconds * _input.blocks_per_second)

    # reference level for dB calculation (adjust based on your needs)
    db_reference: float = 1e-6  # typical digital audio reference

    def __init__(self) -> None:
        self.error: str | None = None

        self._lifecycle: Lifecycle = Lifecycle()

        self._queue: Queue[deque[Block]] | None = None
        self._thread: threading.Thread | None = None

        # locks: acquire in order
        self._levels: Levels | None = None
        self._levels_lock = threading.Lock()

        self._callbacks: dict[str, dict] = {}
        self._callbacks_lock = threading.Lock()

    @property
    def state(self) -> State:
        return self._lifecycle.state

    @property
    def levels(self) -> Levels | None:
        """latest calculated levels (or None if none calculated yet)"""
        with self._levels_lock:
            return self._levels

    def start(self) -> None:
        if not self._lifecycle.prepare_start():
            return
        _logger.info("starting")

        with self._callbacks_lock:
            self._callbacks.clear()

        with self._levels_lock:
            self._levels = None

        self._queue = Queue()

        self._thread = threading.Thread(target=self._worker)
        self._thread.start()

        _input.register_callback(
            f"rms-{id(self)}",
            self._callback,
            window=self.window_blocks,
            hop=self.hop_blocks,
        )

        self._lifecycle.state = State.STARTED

    def stop(self) -> None:
        if not self._lifecycle.prepare_stop():
            return
        _logger.info("stopping")

        _input.remove_callback(f"rms-{id(self)}")

        if self._queue is not None:
            self._queue.put(None)

        if self._thread not in (None, threading.current_thread()):
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                _logger.error("thread failed to stop")

        with self._callbacks_lock:
            self._callbacks.clear()

        with self._levels_lock:
            self._levels = None

        self._queue = None
        self._thread = None

        self._lifecycle.state = State.STOPPED

    def register_callback(
        self,
        name: str,
        callback: Callable[[Levels], None],
    ) -> None:
        """
        register a callback

        args:
        - name: callback name (must be unique)
        - callback: callback function
          - signature: `callback(data: Levels)`
        """
        with self._callbacks_lock:
            self._callbacks[name] = {
                "callback": callback,
            }

    def remove_callback(self, name: str) -> None:
        """
        remove a callback
        """
        with self._callbacks_lock:
            if name in self._callbacks:
                del self._callbacks[name]

    def _callback(self, data: deque[Block]) -> None:
        if self._queue is not None:
            self._queue.put(data)

    def _worker(self) -> None:
        try:
            while True:
                blocks = self._queue.get()

                if blocks is None:
                    self._queue.task_done()
                    return

                # concatenate unfiltered mono audio data for RMS calculation
                # use unfiltered audio to measure overall ambient noise levels
                audio_data = np.concatenate(
                    [block.mono_data.reshape(-1) for block in blocks]
                )

                # calculate RMS
                rms = float(np.sqrt(np.mean(audio_data**2)))

                # convert to decibels (with protection against log(0))
                if rms > 0:
                    db = 20 * math.log10(rms / self.db_reference)
                else:
                    db = float("-inf")  # mathematically correct for zero amplitude

                # use timing from first block
                time = blocks[0].time
                utc = blocks[0].utc

                levels = Levels(
                    rms=rms,
                    db=db,
                    time=time,
                    utc=utc,
                )

                # update stored levels
                with self._levels_lock:
                    self._levels = levels

                # call registered callbacks
                calls = deque()
                with self._callbacks_lock:
                    for c in self._callbacks.values():
                        calls.append(functools.partial(c["callback"], data=levels))

                for c in calls:
                    c()

                self._queue.task_done()

        except Exception:
            self.error = "error in worker"
            _logger.exception(self.error)
            self.stop()
