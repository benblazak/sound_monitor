import csv
import logging
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
from sound_monitor.util.types.singleton import Singleton

_config = Config.get()
_logger = logging.getLogger(__name__)

_input = Input.get()


class _Block:
    def __init__(self, data: deque[Block]):
        self.data = np.concatenate([block.yamnet_data.reshape(-1) for block in data])
        self.time = data[0].time
        self.clock = data[0].clock


class Scores:
    @staticmethod
    def _get_class_names() -> list[str]:
        with open(_config.yamnet_class_map, encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)  # skip header
            return [row[2] for row in reader]

    class_names = _get_class_names()

    @classmethod
    def max(cls, *scores: Self) -> Self:
        "max scores -- time and clock are from the first score"
        data = np.max([score.data for score in scores], axis=0)
        return cls(
            data=data,
            time=scores[0].time,
            clock=scores[0].clock,
        )

    @classmethod
    def mean(cls, *scores: Self) -> Self:
        "mean scores -- time and clock are from the first score"
        data = np.mean([score.data for score in scores], axis=0)
        return cls(
            data=data,
            time=scores[0].time,
            clock=scores[0].clock,
        )

    def __init__(self, data: np.ndarray, time: float, clock: datetime) -> None:
        self.data: np.ndarray = data
        """
        scores -- float32, shape (512,)

        values are between 0 and 1, representing confidence scores for each class

        https://www.kaggle.com/models/google/yamnet/tfLite/tflite
        """

        self.time: float = time
        self.clock: datetime = clock


class YAMNet(Singleton["YAMNet"]):
    if Input.blocks_per_second != 10:
        raise ValueError("YAMNet depends on 0.1s blocks")

    sample_rate = 16000  # native
    window_seconds = 0.96  # native
    hop_seconds = 0.5  # native is 0.48s, but we need to align with Input blocks

    window_samples = int(window_seconds * sample_rate)
    hop_samples = int(hop_seconds * sample_rate)
    buffer_samples = sample_rate  # 1s buffer
    input_interval = int(hop_seconds * _input.blocks_per_second)

    def __init__(self) -> None:
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

        self._callbacks: dict[str, dict] = {}
        self._callbacks_lock = threading.Lock()

        # lock for writes (in this file), or to (very quickly) pause writes
        self._scores: deque[Scores] = deque(maxlen=2)
        self._scores_lock = threading.Lock()

        self._queue: Queue[_Block] | None = None
        self._thread: threading.Thread | None = None
        self._last_block: _Block | None = None

    @property
    def time(self) -> float:
        return self._scores[-1].time

    @property
    def clock(self) -> datetime:
        return self._scores[-1].clock

    @property
    def scores(self) -> Scores:
        """
        latest scores

        covering `time` to `time + 1s`

        actually from `time` to `time + 0.96s`, but we'll treat it like a full
        second
        """
        return self._scores[-1]

    @property
    def scores_max(self) -> Scores:
        """ "
        max scores

        covering `time` to `time + 0.5s`

        actually from `time` to `time + 0.46s`, but we'll treat it like a full
        half second

        each window overlaps with the previous window by about 0.5s. this is the
        max of the scores for the two windows that overlap from `time` to
        `time + 0.5s`
        """
        return Scores.max(*reversed(self._scores))

    @property
    def scores_mean(self) -> Scores:
        """
        mean scores

        actually from `time` to `time + 0.46s`, but we'll treat it like a full
        half second

        each window overlaps with the previous window by about 0.5s. this is the
        mean of the scores for the two windows that overlap from `time` to
        `time + 0.5s`
        """
        return Scores.mean(*reversed(self._scores))

    def start(self) -> None:
        if self._queue is not None:
            return
        self._queue = Queue()

        _input.register_callback(
            f"yamnet-{id(self)}",
            self._callback,
            interval=self.input_interval,
        )

        self._thread = threading.Thread(target=self._worker)
        self._thread.start()

    def stop(self) -> None:
        _input.remove_callback(f"yamnet-{id(self)}")

        if self._queue is not None:
            self._queue.put(None)

        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                _logger.error("thread failed to stop")

        self._queue = None
        self._thread = None
        self._last_block = None

    def register_callback(
        self,
        name: str,
        callback: Callable[["YAMNet"], None],
    ) -> None:
        with self._callbacks_lock:
            self._callbacks[name] = {
                "callback": callback,
            }

    def remove_callback(self, name: str) -> None:
        with self._callbacks_lock:
            if name in self._callbacks:
                del self._callbacks[name]

    def _callback(self, input: Input) -> None:
        if self._queue is not None:
            blocks = input.last_blocks(self.input_interval)
            self._queue.put(_Block(blocks))

    def _worker(self) -> None:
        try:
            while True:
                block = self._queue.get()
                self._queue.task_done()

                if block is None:
                    return

                if self._last_block is None:
                    self._last_block = block
                    continue

                data = np.concatenate([self._last_block.data, block.data])[
                    : self.window_samples
                ]
                time = self._last_block.time
                clock = self._last_block.clock

                self._interpreter.set_tensor(
                    self._waveform_input_index,
                    data,
                )
                self._interpreter.invoke()

                scores: np.ndarray = self._interpreter.get_tensor(
                    self._scores_output_index
                )
                scores = scores.reshape(-1)  # reshape from (1, 512) to (512,)

                with self._scores_lock:
                    self._scores.append(
                        Scores(
                            data=scores,
                            time=time,
                            clock=clock,
                        )
                    )

                callbacks = []
                with self._callbacks_lock:
                    for c in self._callbacks.values():
                        callbacks.append(c)
                for c in callbacks:
                    c["callback"](yamnet=self)

        except Exception as e:
            _logger.error(f"error in worker: {e}")
            self.stop()
