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
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, peak_widths

from sound_monitor.audio.direction import Direction, find_direction
from sound_monitor.audio.input import Block, Input
from sound_monitor.config import Config
from sound_monitor.util.types.lifecycle import Lifecycle, State
from sound_monitor.util.types.singleton import Singleton

_config = Config.get()
_logger = logging.getLogger(__name__)

_input = Input.get()


class Event:
    """
    A detected barking event (or other loud sound).
    
    This represents a single sound that was loud enough and the right 
    duration to be considered interesting (like a dog bark).
    """

    def __init__(
        self,
        amplitude: float,
        width: float,
        time: float,
        utc: datetime,
        direction: Direction,
    ) -> None:
        self.amplitude: float = amplitude
        """how loud the peak was (higher = louder)"""

        self.width: float = width
        """how long the sound lasted in seconds (e.g., 0.1s for a bark)"""

        self.time: float = time
        """stream time when the peak occurred"""

        self.utc: datetime = utc
        """utc time when the peak occurred"""
        
        self.direction: Direction = direction
        """direction the sound came from (xyz unit vector + confidence)"""

    @property
    def clock(self) -> datetime:
        """local time when the peak occurred"""
        return self.utc.astimezone(_config.timezone)


class Peak(Singleton["Peak"]):
    """
    Real-time peak detector for barking events.
    
    HOW IT WORKS (in simple terms):
    1. Input provides 0.5-second windows, sliding every 0.3 seconds
    2. Smooths the audio to create an "envelope" (like drawing a line over the peaks)
    3. Learns what "background noise" sounds like over time
    4. Detects sounds that are much louder than background AND the right duration for barks
    5. Reports these as "events"
    
    WHY THE SLIDING WINDOWS:
    - 0.5s window: Long enough to capture a full bark
    - 0.3s hop: Short enough to not miss rapid barks
    - Input handles windowing automatically (no manual state management needed)
    """

    # timing parameters - how we slice up the audio stream
    window_seconds: float = 0.50    # analyze 0.5s of audio at a time
    hop_seconds: float = 0.30       # move forward 0.3s between analyses
    
    # sound characteristics - what we're looking for
    bark_min_width: float = 0.07    # barks are at least 70ms long
    bark_max_width: float = 0.15    # barks are at most 150ms long
    envelope_smooth: float = 0.10   # smooth audio over 100ms to create envelope
    
    # adaptive threshold parameters - how we learn background noise
    noise_adapt_seconds: float = getattr(_config, 'peak_threshold_smooth_seconds', 10.0)
    """how quickly we adapt to background noise changes (10s = slow adaptation)"""
    
    threshold_multiplier: float = getattr(_config, 'peak_threshold_k_mad', 4.0)
    """how much louder than background noise a sound must be to count as an event"""

    # convert timing to samples
    sample_rate = _config.uma8_sample_rate
    window_samples = int(window_seconds * sample_rate)
    envelope_samples = int(envelope_smooth * sample_rate)
    
    # convert timing to input blocks (input gives us ~10 blocks per second)
    window_blocks = math.ceil(window_seconds * _input.blocks_per_second)
    hop_blocks = round(hop_seconds * _input.blocks_per_second)
    
    # valid detection region within each window
    # Input provides overlapping windows, so we only report peaks from the middle section
    # to avoid double-reporting the same peak from adjacent windows
    hop_samples = int(hop_seconds * sample_rate)
    context_samples = (window_samples - hop_samples) // 2  # context on each side
    valid_start = context_samples
    valid_end = context_samples + hop_samples
    
    # minimum distance between detected peaks (prevents double-detection)
    min_peak_separation = envelope_samples // 2

    def __init__(self) -> None:
        self.error: str | None = None

        self._lifecycle: Lifecycle = Lifecycle()

        self._queue: Queue[deque[Block]] | None = None
        self._thread: threading.Thread | None = None
        
        # adaptive noise floor tracking
        # MAD = "Median Absolute Deviation" - a robust way to measure noise variation
        # Think of it like standard deviation, but less affected by occasional loud sounds
        self._noise_median: float | None = None     # typical background noise level
        self._noise_mad: float | None = None        # how much background noise varies
        self._noise_alpha = self.hop_seconds / self.noise_adapt_seconds  # adaptation speed

        # results storage
        self._latest_event: Event | None = None
        self._event_lock = threading.Lock()

        # callback management
        self._callbacks: dict[str, dict] = {}
        self._callbacks_lock = threading.Lock()

    @property
    def state(self) -> State:
        return self._lifecycle.state

    @property
    def event(self) -> Event | None:
        """latest detected event (or None if no events detected yet)"""
        with self._event_lock:
            return self._latest_event

    def start(self) -> None:
        if not self._lifecycle.prepare_start():
            return
        _logger.info("starting")

        with self._callbacks_lock:
            self._callbacks.clear()

        with self._event_lock:
            self._latest_event = None

        # reset detection state
        self._noise_median = None
        self._noise_mad = None

        self._queue = Queue()

        self._thread = threading.Thread(target=self._worker)
        self._thread.start()

        _input.register_callback(
            f"peak-{id(self)}",
            self._callback,
            window=self.window_blocks,
            hop=self.hop_blocks,
        )

        self._lifecycle.state = State.STARTED

    def stop(self) -> None:
        if not self._lifecycle.prepare_stop():
            return
        _logger.info("stopping")

        _input.remove_callback(f"peak-{id(self)}")

        if self._queue is not None:
            self._queue.put(None)

        if self._thread not in (None, threading.current_thread()):
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                _logger.error("thread failed to stop")

        with self._callbacks_lock:
            self._callbacks.clear()

        with self._event_lock:
            self._latest_event = None

        # reset detection state
        self._noise_median = None
        self._noise_mad = None

        self._queue = None
        self._thread = None

        self._lifecycle.state = State.STOPPED

    def register_callback(
        self,
        name: str,
        callback: Callable[[Event], None],
    ) -> None:
        """
        register a callback for when events are detected

        args:
        - name: callback name (must be unique)
        - callback: callback function
          - signature: `callback(data: Event)`
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

                self._process_audio_blocks(blocks)
                self._queue.task_done()

        except Exception:
            self.error = "error in worker"
            _logger.exception(self.error)
            self.stop()

    def _process_audio_blocks(self, blocks: deque[Block]) -> None:
        """
        Main peak detection algorithm.
        
        STEP BY STEP:
        1. Concatenate audio from blocks (Input handles windowing)
        2. Create a smooth envelope to find the overall loudness over time
        3. Learn what the background noise looks like
        4. Find peaks that are much louder than background and bark-shaped
        5. Report any valid events found
        """
        
        # STEP 1: Get audio data from Input's windowing
        # Input provides exactly the right blocks for our window
        window_audio = np.concatenate(
            [block.peak_data.reshape(-1) for block in blocks]
        )[:self.window_samples]  # trim to exact window size
        
        window_multichannel = np.concatenate(
            [block.direction_data for block in blocks]
        )[:self.window_samples]  # trim to exact window size

        # STEP 2: Create smooth envelope
        # RMS envelope = "how loud is the audio at each moment"
        # We square the audio (makes positive), smooth it, then take square root
        envelope = np.sqrt(
            uniform_filter1d(
                window_audio.astype(np.float32) ** 2,
                size=self.envelope_samples,
                mode="constant",
            )
        )

        # STEP 3: Learn background noise characteristics
        # We use median (middle value) instead of average because it ignores loud sounds
        current_median = float(np.median(envelope))
        
        # MAD = Median Absolute Deviation = how spread out the noise is
        # It's like standard deviation but more robust to outliers
        current_mad = float(np.median(np.abs(envelope - current_median)))
        
        # Slowly adapt our noise estimates (exponential smoothing)
        if self._noise_median is None:
            # First time - just use current values
            self._noise_median = current_median
            self._noise_mad = current_mad
        else:
            # Update gradually - mostly keep old estimate, blend in new
            self._noise_median = (1 - self._noise_alpha) * self._noise_median + self._noise_alpha * current_median
            self._noise_mad = (1 - self._noise_alpha) * self._noise_mad + self._noise_alpha * current_mad

        # Set detection threshold: background + (variation × multiplier)
        # A sound must be this loud to be considered an event
        detection_threshold = self._noise_median + self.threshold_multiplier * self._noise_mad

        # STEP 4: Find peaks that look like barks
        peaks, properties = find_peaks(
            envelope,
            height=detection_threshold,                                          # must be loud enough
            distance=self.min_peak_separation,                                   # can't be too close together
            width=(
                int(self.bark_min_width * self.sample_rate),                    # must be at least bark_min_width
                int(self.bark_max_width * self.sample_rate)                     # must be at most bark_max_width
            ),
        )

        # STEP 5: Filter to valid detection region
        # Only report peaks from the middle section to avoid double-reporting
        # between adjacent windows
        valid_mask = (peaks >= self.valid_start) & (peaks < self.valid_end)
        valid_peaks = peaks[valid_mask]
        
        if len(valid_peaks) == 0:
            # No events found
            return

        # Calculate peak characteristics
        peak_widths_seconds = (
            peak_widths(envelope, valid_peaks, rel_height=0.5)[0] / self.sample_rate
        )
        peak_amplitudes = properties["peak_heights"][valid_mask]

        # Create Event objects with direction information
        events = []
        for peak_idx, amplitude, width in zip(valid_peaks, peak_amplitudes, peak_widths_seconds):
            # Calculate when this peak happened in absolute time
            # peak_idx is position in window, blocks[0].time is start of window
            time_in_window = peak_idx / self.sample_rate
            absolute_time = blocks[0].time + time_in_window
            
            # Extract audio around this peak for direction finding
            # Use the actual detected peak width, but ensure minimum size for good correlation
            width_samples = int(width * self.sample_rate)
            direction_window_samples = max(width_samples * 2, self.envelope_samples)  # at least 2x peak width or 100ms
            
            peak_start = max(0, peak_idx - direction_window_samples // 2)
            peak_end = min(len(window_multichannel), peak_idx + direction_window_samples // 2)
            
            # Extract multichannel audio for direction finding
            if peak_end > peak_start:
                peak_multichannel_audio = window_multichannel[peak_start:peak_end]
                direction = find_direction(peak_multichannel_audio)
                if direction is None:
                    # Direction calculation failed - skip this event
                    continue
            else:
                # Can't extract audio - skip this event
                continue
            
            event = Event(
                amplitude=float(amplitude),
                width=float(width),
                time=absolute_time,
                utc=blocks[0].utc,
                direction=direction,
            )
            events.append(event)

        if events:
            # Keep the loudest event (for now - could report all)
            loudest_event = max(events, key=lambda e: e.amplitude)
            
            # Store latest event
            with self._event_lock:
                self._latest_event = loudest_event

            # Notify callbacks
            calls = deque()
            with self._callbacks_lock:
                for callback_info in self._callbacks.values():
                    calls.append(functools.partial(callback_info["callback"], data=loudest_event))

            for call in calls:
                call()
