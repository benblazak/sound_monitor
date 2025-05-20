import threading
from enum import StrEnum, auto
from typing import Self


class State(StrEnum):
    STARTING = auto()
    STARTED = auto()
    STOPPING = auto()
    STOPPED = auto()


class Lifecycle:
    def __init__(self, state: State = State.STOPPED) -> None:
        self._state = state
        self._condition = threading.Condition()

    @property
    def state(self) -> State:
        with self._condition:
            return self._state

    @state.setter
    def state(self, state: State) -> None:
        with self._condition:
            self._state = state
            self._condition.notify_all()

    def __enter__(self) -> Self:
        self._condition.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._condition.notify_all()
        self._condition.release()

    @property
    def wait_for(self):
        return self._condition.wait_for

    def wait_for_state(self, state: State, timeout: float | None = None) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: self._state == state, timeout=timeout
            )

    def prepare_start(self) -> bool:
        """
        prepare to start

        returns:
        - True if the caller should start
          - the caller should set state to State.STARTED when done
        - False if the caller should return

        raises:
        - RuntimeError if currently stopping
        """
        with self._condition:
            if self._state in (State.STARTING, State.STARTED):
                return False
            elif self._state is State.STOPPING:
                raise RuntimeError("can't start: currently stopping")
            self._state = State.STARTING
            return True

    def prepare_stop(self) -> bool:
        """
        prepare to stop

        returns:
        - True if the caller should stop
          - the caller should set state to State.STOPPED when done
        - False if the caller should return

        raises:
        - RuntimeError if currently starting
        """
        with self._condition:
            if self._state in (State.STOPPING, State.STOPPED):
                return False
            elif self._state is State.STARTING:
                raise RuntimeError("can't stop: currently starting")
            self._state = State.STOPPING
            return True
