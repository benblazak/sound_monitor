import csv
import functools
import logging
import math
import threading
from collections import deque
from collections.abc import Callable
from datetime import datetime
from queue import Queue
from typing import Self

import numpy as np
from ai_edge_litert.interpreter import Interpreter

from sound_monitor.audio.input import Block, Input
from sound_monitor.config import Config
from sound_monitor.util.types.lifecycle import Lifecycle, State
from sound_monitor.util.types.singleton import Singleton

_config = Config.get()
_logger = logging.getLogger(__name__)

_input = Input.get()


class Scores:

    @staticmethod
    def _get_class_names() -> np.ndarray:
        """get class names from yamnet class map"""
        with open(_config.yamnet_class_map, encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)  # skip header
            return np.array([row[2].lower() for row in reader], dtype=str)

    class_names = _get_class_names()

    def __init__(
        self,
        data: np.ndarray,
        time: float,
        clock: datetime,
        *,
        previous: Self | None = None,
    ) -> None:

        self.data: np.ndarray = data
        """
        scores -- float32, shape (512,)

        values are between 0 and 1, representing confidence scores for each class

        https://www.kaggle.com/models/google/yamnet/tfLite/tflite
        """

        self.time: float = time
        self.clock: datetime = clock

        self.previous: Self | None = (
            None
            if previous is None
            else self.__class__(
                data=previous.data,
                time=previous.time,
                clock=previous.clock,
            )
        )
        """
        previous scores -- for calculating max and mean

        notes:
        - we make a (shallow) copy without previous.previous to avoid creating
          an implicit linked list
        """

    @property
    def raw(self) -> dict[str, float]:
        """
        raw scores

        covering `time` to `time + 1s`

        actually from `time` to `time + 0.96s`, but we'll treat it like a full
        second
        """
        return dict(
            zip(
                self.class_names,
                self.data,
            )
        )

    @property
    def max(self) -> dict[str, float]:
        """
        max scores

        covering `time` to `time + 0.5s`

        actually from `time` to `time + 0.46s`, but we'll treat it like a full
        half second

        each window overlaps with the previous window by about 0.5s. this is the
        max of the scores for the two windows that overlap from `time` to
        `time + 0.5s`
        """
        if self.previous is None:
            return self.raw
        return dict(
            zip(
                self.class_names,
                np.max([self.previous.data, self.data], axis=0),
            )
        )

    @property
    def mean(self) -> dict[str, float]:
        """
        mean scores

        covering `time` to `time + 0.5s`

        actually from `time` to `time + 0.46s`, but we'll treat it like a full
        half second

        each window overlaps with the previous window by about 0.5s. this is the
        mean of the scores for the two windows that overlap from `time` to
        `time + 0.5s`
        """
        if self.previous is None:
            return self.raw
        return dict(
            zip(
                self.class_names,
                np.mean([self.previous.data, self.data], axis=0),
            )
        )


class YAMNet(Singleton["YAMNet"]):

    sample_rate = 16000  # native
    window_seconds = 0.96  # native
    hop_seconds = 0.48  # native

    window_samples = int(window_seconds * sample_rate)

    window_blocks = math.ceil(window_seconds * _input.blocks_per_second)
    hop_blocks = round(hop_seconds * _input.blocks_per_second)
    if _input.blocks_per_second != 10:
        _logger.warning(
            "check comments and assumptions\n"
            f"  - blocks_per_second: {_input.blocks_per_second}\n"
            f"  - window_blocks: {window_blocks}\n"
            f"  - hop_blocks: {hop_blocks}"
        )

    def __init__(self) -> None:
        self.error: str | None = None

        self._interpreter = Interpreter(model_path=str(_config.yamnet_model))

        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        self._waveform_input_index = input_details[0]["index"]
        self._scores_output_index = output_details[0]["index"]

        self._interpreter.resize_tensor_input(
            self._waveform_input_index,
            [self.window_samples],
            strict=True,
        )
        self._interpreter.allocate_tensors()

        self._lifecycle: Lifecycle = Lifecycle()

        self._queue: Queue[deque[Block]] | None = None
        self._thread: threading.Thread | None = None
        self._last_scores: Scores | None = None

        # locks: acquire in order

        self._callbacks: dict[str, dict] = {}
        self._callbacks_lock = threading.Lock()

    def start(self) -> None:
        if not self._lifecycle.prepare_start():
            return

        with self._callbacks_lock:
            self._callbacks.clear()

        self._queue = Queue()

        self._thread = threading.Thread(target=self._worker)
        self._thread.start()

        _input.register_callback(
            f"yamnet-{id(self)}",
            self._callback,
            window=self.window_blocks,
            hop=self.hop_blocks,
        )

        self._lifecycle.state = State.STARTED

    def stop(self) -> None:
        if not self._lifecycle.prepare_stop():
            return

        _input.remove_callback(f"yamnet-{id(self)}")

        if self._queue is not None:
            self._queue.put(None)

        if self._thread not in (None, threading.current_thread()):
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                _logger.error("thread failed to stop")

        with self._callbacks_lock:
            self._callbacks.clear()

        self._queue = None
        self._thread = None
        self._last_scores = None

        self._lifecycle.state = State.STOPPED

    def register_callback(
        self,
        name: str,
        callback: Callable[[Scores], None],
    ) -> None:
        """
        register a callback

        args:
        - name: callback name (must be unique)
        - callback: callback function
          - signature: `callback(data: Scores)`
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

                data = np.concatenate(
                    [block.yamnet_data.reshape(-1) for block in blocks]
                )[: self.window_samples]
                time = blocks[0].time
                clock = blocks[0].clock

                self._interpreter.set_tensor(
                    self._waveform_input_index,
                    data,
                )
                self._interpreter.invoke()

                scores: np.ndarray = self._interpreter.get_tensor(
                    self._scores_output_index,
                )
                scores = scores.reshape(-1)  # reshape from (1, 512) to (512,)

                self._last_scores = Scores(
                    data=scores,
                    time=time,
                    clock=clock,
                    previous=self._last_scores,
                )

                calls = deque()

                with self._callbacks_lock:
                    for c in self._callbacks.values():
                        calls.append(
                            functools.partial(
                                c["callback"],
                                data=self._last_scores,
                            )
                        )

                for c in calls:
                    c()

                self._queue.task_done()

        except Exception:
            self.error = "error in worker"
            _logger.exception(self.error)
            self.stop()
