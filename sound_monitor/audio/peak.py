import logging
from collections import deque
from queue import Queue

import numpy as np

from sound_monitor.audio.input import Block, Input
from sound_monitor.config import Config

_config = Config.get()
_logger = logging.getLogger(__name__)


class _Block:
    def __init__(self, data: deque[Block]) -> None:
        self.raw = data
        self.data = np.concatenate([block.yamnet_data.reshape(-1) for block in data])
        self.time = data[0].time
        self.clock = data[0].clock


class Event:
    def __init__(self) -> None:
        pass


class Peak:

    def __init__(self) -> None:
        self._buffer: deque[Block] | None = None
        self._queue: Queue[_Block] | None = None

    def _callback(self, input: Input) -> None:
        if self._queue is not None:
            pass
