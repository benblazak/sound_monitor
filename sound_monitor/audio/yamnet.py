import csv
import functools
import logging
import math
import threading
from collections import OrderedDict, deque
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

    @classmethod
    def max(cls, scores: list[Self], *, time_index: int = 0) -> Self:
        """
        max scores

        args:
        - scores: the scores to aggregate
        - time_index: the index of the score to use for the time

        notes:
        - assuming a sorted `scores`, `time_index` should probably be
          - 0 if the scores don't overlap in duration
          - 1 if the scores overlap in duration
          - this is because if the scores overlap, the time of the aggregate
            should be the beginning of the overlapping period

        returns:
        - a `Scores` object with the aggregate data

        raises:
        - ValueError: if no scores are provided
        """
        if len(scores) == 0:
            raise ValueError("no scores")
        if len(scores) == 1:
            return scores[0]

        start = scores[time_index]
        return cls(
            data=np.max([s.data for s in scores], axis=0),
            time=start.time,
            utc=start.utc,
        )

    @classmethod
    def mean(cls, scores: list[Self], *, time_index: int = 0) -> Self:
        """
        mean scores

        args:
        - scores: the scores to aggregate
        - time_index: the index of the score to use for the time

        notes:
        - assuming a sorted `scores`, `time_index` should probably be
          - 0 if the scores don't overlap in duration
          - 1 if the scores overlap in duration
          - this is because if the scores overlap, the time of the aggregate
            should be the beginning of the overlapping period

        returns:
        - a `Scores` object with the aggregate data

        raises:
        - ValueError: if no scores are provided
        """
        if len(scores) == 0:
            raise ValueError("no scores")
        if len(scores) == 1:
            return scores[0]

        start = scores[time_index]
        return cls(
            data=np.mean([s.data for s in scores], axis=0),
            time=start.time,
            utc=start.utc,
        )

    @classmethod
    def percentile(
        cls, scores: list[Self], *, percentile: float, time_index: int = 0
    ) -> Self:
        """
        percentile scores

        args:
        - scores: the scores to aggregate
        - time_index: the index of the score to use for the time
        - percentile: the percentile to use

        notes:
        - assuming a sorted `scores`, `time_index` should probably be
          - 0 if the scores don't overlap in duration
          - 1 if the scores overlap in duration
          - this is because if the scores overlap, the time of the aggregate
            should be the beginning of the overlapping period

        returns:
        - a `Scores` object with the aggregate data

        raises:
        - ValueError: if no scores are provided
        """
        if len(scores) == 0:
            raise ValueError("no scores")
        if len(scores) == 1:
            return scores[0]

        start = scores[time_index]
        return cls(
            data=np.percentile([s.data for s in scores], axis=0, q=percentile),
            time=start.time,
            utc=start.utc,
        )

    def __init__(
        self,
        data: np.ndarray,
        time: float,
        utc: datetime,
    ) -> None:

        self.data: np.ndarray = data
        """
        scores -- float32, shape (512,)

        values are between 0 and 1, representing confidence scores for each class

        https://www.kaggle.com/models/google/yamnet/tfLite/tflite
        """

        self.time: float = time
        self.utc: datetime = utc

    def to_dict(
        self,
        *,
        top: int | None = None,
        threshold: float | None = None,
    ) -> OrderedDict[str, float]:
        """
        convert to a dictionary

        args:
        - top: include at most this many entries, sorted by score, descending
        - threshold: filter scores below this value

        returns:
        - a ordered dictionary (class name -> score)

        notes:
        - may have fewer than `top` scores if `threshold` is set
        - scores are in class index order, unless `top` is set in which case
          they are sorted by score, descending
        """
        indices = np.arange(len(self.data))

        if top is not None:
            indices = np.argsort(self.data)[-top:][::-1]
        if threshold is not None:
            indices = indices[self.data[indices] >= threshold]

        return OrderedDict(zip(self.class_names[indices], self.data[indices]))


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

        # locks: acquire in order

        self._callbacks: dict[str, dict] = {}
        self._callbacks_lock = threading.Lock()

    def start(self) -> None:
        if not self._lifecycle.prepare_start():
            return
        _logger.info("starting")

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
        _logger.info("stopping")

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

                audio_data = np.concatenate(
                    [block.yamnet_data.reshape(-1) for block in blocks]
                )[: self.window_samples]
                time = blocks[0].time
                utc = blocks[0].utc

                self._interpreter.set_tensor(
                    self._waveform_input_index,
                    audio_data,
                )
                self._interpreter.invoke()

                scores_data: np.ndarray = self._interpreter.get_tensor(
                    self._scores_output_index,
                )
                scores_data = scores_data.reshape(-1)  # reshape from (1, 512) to (512,)

                scores = Scores(
                    data=scores_data,
                    time=time,
                    utc=utc,
                )

                calls = deque()

                with self._callbacks_lock:
                    for c in self._callbacks.values():
                        calls.append(functools.partial(c["callback"], data=scores))

                for c in calls:
                    c()

                self._queue.task_done()

        except Exception:
            self.error = "error in worker"
            _logger.exception(self.error)
            self.stop()
