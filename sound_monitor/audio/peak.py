"""sound_monitor.audio.peak

Real-time peak detector for barking events.

Algorithm (plain English)
-------------------------
1. The microphone delivers 0.10 s *blocks* (frames of raw PCM).
2. We analyse the stream in overlapping 0.50 s *windows* that slide by
   0.30 s – every analysis step therefore sees:
       0.10 s   left-hand **context** (ignored when reporting)
       0.30 s   **valid region** where peaks are accepted
       0.10 s   right-hand **context** (guarantees full coverage)
3. Inside each window we build an RMS envelope (10 ms smoothing) and let
   ``scipy.signal.find_peaks`` locate peaks whose width fits a bark
   (70–150 ms) and whose height stands **k·MAD** above the running median
   of the noise floor.
4. A simple exponential smoother with time-constant
   ``THRESHOLD_SMOOTH_SEC`` (≈10 s) updates the median and MAD, so the
   threshold tracks slow background changes but ignores short-lived
   peaks.

Tunable parameters
------------------
THRESHOLD_SMOOTH_SEC – how fast the noise floor estimate adapts.  Lower
values follow ambience changes faster but risk reacting to bursts.
THRESHOLD_K_MAD       – how many "robust sigmas" above the noise floor a
peak must rise to be reported.

These two parameters affect *what* gets reported, so they are exposed in
``sound_monitor.config``; everything else is implementation detail.
"""

# TODO this is mostly from o3, need to go through it, understand it, and then rewrite :)

import logging
from collections import deque
from datetime import datetime
from queue import Queue
from threading import Lock, Thread

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, peak_widths

from sound_monitor.audio.input import Block, Input
from sound_monitor.config import Config
from sound_monitor.util.types.singleton import Singleton

_config = Config.get()
_logger = logging.getLogger(__name__)

_input = Input.get()  # reuse the singleton like yamnet.py


class _Block:
    def __init__(self, data: deque[Block]) -> None:
        # use filtered mono signal for peak detection
        self.raw = data
        self.data = np.concatenate([block.peak_data.reshape(-1) for block in data])
        self.time = data[0].time  # stream time of first sample in this hop
        self.clock = data[0].clock


class Event:
    """One detected peak inside a hop.

    Parameters
    ----------
    time : float
        Stream time of the first sample of the peak centre.
    clock : datetime
        Wall-clock time that corresponds to ``time``.
    amplitude : float
        Envelope height at the detected peak.
    width : float
        Width estimate in **seconds** (full width at half prominence).
    """

    def __init__(self, time: float, clock: datetime, amplitude: float, width: float):
        self.time = time
        self.clock = clock
        self.amplitude = amplitude
        self.width = width


class Peak(Singleton["Peak"]):

    # --- signal-processing constants -------------------------------------------------
    sample_rate = _config.uma8_sample_rate
    window_seconds = 0.50
    hop_seconds = 0.30
    context_seconds = 0.10  # on each side
    peak_width_seconds = 0.10
    THRESHOLD_SMOOTH_SEC = _config.peak_threshold_smooth_seconds  # ≈10 s
    THRESHOLD_K_MAD = _config.peak_threshold_k_mad  # e.g. 4

    window_samples = int(window_seconds * sample_rate)
    hop_samples = int(hop_seconds * sample_rate)
    context_samples = int(context_seconds * sample_rate)
    valid_left = context_samples
    valid_right = context_samples + hop_samples  # 0.10s+0.30s = 0.40s (exclusive)

    peak_width_samples = int(peak_width_seconds * sample_rate)
    distance_samples = peak_width_samples // 2  # ensure separation of detected peaks

    # how many 0.1-s Input blocks make up one hop (should be 3)
    input_interval = int(hop_seconds * _input.blocks_per_second)

    def __init__(self) -> None:
        self._queue: Queue[_Block] | None = None
        self._thread: Thread | None = None

        # detection state
        self._tail = np.zeros(self.window_samples - self.hop_samples, dtype=np.float32)
        self._med: float | None = None
        self._mad: float | None = None
        self._alpha = self.hop_seconds / self.THRESHOLD_SMOOTH_SEC

        # results + callbacks
        self._event: Event | None = None
        self._event_lock = Lock()
        self._callbacks: dict[str, dict] = {}
        self._callbacks_lock = Lock()

    # --------------------------------------------------------------------- public ---
    def start(self) -> None:
        if self._queue is not None:
            return  # already running
        self._queue = Queue()
        _input.register_callback(
            f"peak-{id(self)}", self._callback, interval=self.input_interval
        )
        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        _input.remove_callback(f"peak-{id(self)}")
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                _logger.error("peak worker failed to stop")
        self._queue = None
        self._thread = None
        self._tail = np.zeros_like(self._tail)
        self._med = self._mad = None

    # ---------------- YAMNet-style public interface -------------------
    @property
    def event(self) -> Event | None:
        """Latest detected peak (or *None* if none seen yet)."""
        with self._event_lock:
            return self._event

    @property
    def time(self) -> float:
        return self._event.time if self._event else _input.time

    @property
    def clock(self):
        return self._event.clock if self._event else _input.clock

    # callbacks ---------------------------------------------------------------------
    def register_callback(self, name: str, callback):
        with self._callbacks_lock:
            self._callbacks[name] = {"callback": callback}

    def remove_callback(self, name: str):
        with self._callbacks_lock:
            if name in self._callbacks:
                del self._callbacks[name]

    # ------------------------------------------------------------------ internals ---
    def _callback(self, input: Input) -> None:
        if self._queue is None:
            return
        blocks = input.last_blocks(self.input_interval)
        self._queue.put(_Block(blocks))

    def _worker(self) -> None:
        try:
            while True:
                blk = self._queue.get()
                self._queue.task_done()
                if blk is None:
                    return
                self._process_block(blk)
        except Exception:
            _logger.exception("error in peak worker")
            self.stop()

    def _process_block(self, blk: _Block) -> None:
        # warm-up: need one full hop before we have real context
        if not hasattr(self, "_warm"):
            self._tail = blk.data[-self._tail.shape[0] :]
            self._warm = True
            return

        # 1. build analysis frame (tail + new hop)
        frame = np.concatenate([self._tail, blk.data])
        if frame.shape[0] < self.window_samples:
            pad = np.zeros(self.window_samples - frame.shape[0], dtype=frame.dtype)
            frame = np.concatenate([frame, pad])
        else:
            frame = frame[: self.window_samples]

        # 2. RMS envelope
        env = np.sqrt(
            uniform_filter1d(
                frame.astype(np.float32) ** 2,
                size=self.peak_width_samples,
                mode="constant",
            )
        )

        # 3. update noise statistics (median + MAD)
        m_win = float(np.median(env))
        d_win = float(np.median(np.abs(env - m_win)))
        if self._med is None:
            self._med, self._mad = m_win, d_win
        else:
            self._med = (1 - self._alpha) * self._med + self._alpha * m_win
            self._mad = (1 - self._alpha) * self._mad + self._alpha * d_win

        threshold = self._med + self.THRESHOLD_K_MAD * self._mad

        # 4. peak detection
        peaks, props = find_peaks(
            env,
            height=threshold,
            distance=self.distance_samples,
            width=(int(0.07 * self.sample_rate), int(0.15 * self.sample_rate)),
        )

        # 5. filter to valid region
        mask = (peaks >= self.valid_left) & (peaks < self.valid_right)
        peaks = peaks[mask]
        if len(peaks) == 0:
            self._roll_tail(frame)
            return

        widths = (
            peak_widths(env, peaks, rel_height=0.5)[0] / self.sample_rate
        )  # seconds
        heights = props["peak_heights"][mask]

        # 6. emit events
        events: list[Event] = []
        for idx, amp, wid in zip(peaks, heights, widths):
            # estimate absolute time of peak centre
            t_rel = (idx - self.context_samples) / self.sample_rate
            evt_time = blk.time + t_rel - (self.window_seconds - self.hop_seconds)
            events.append(Event(evt_time, blk.clock, float(amp), float(wid)))

        if events:
            # keep loudest event only (for now)
            evt = max(events, key=lambda e: e.amplitude)
            with self._event_lock:
                self._event = evt
            callbacks = []
            with self._callbacks_lock:
                callbacks = [c for c in self._callbacks.values()]
            for c in callbacks:
                c["callback"](peak=self, events=events)

        # 7. roll tail for next frame
        self._roll_tail(frame)

    def _roll_tail(self, frame: np.ndarray) -> None:
        # keep rightmost window-hop samples
        self._tail = frame[self.hop_samples :]
