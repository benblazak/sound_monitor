import threading
from enum import StrEnum, auto
from typing import Self


class State(StrEnum):
    INIT = auto()
    STARTING = auto()
    STARTED = auto()
    STOPPING = auto()
    STOPPED = auto()
    DONE = auto()


class InitStopBehavior(StrEnum):
    ALLOW = auto()
    NOP = auto()
    RAISE = auto()


class StateTransitionError(Exception):
    pass


class Lifecycle:
    """
    lifecycle management

    typical state transitions:
    ```text
    INIT -------> STARTING -----> STARTED
     |               ^               |
     |               |               v
     |            STOPPED <----- STOPPING
     |                               |
     v                               v
    DONE <---------------------------+
    ```
    """

    def __init__(self, state: State = State.INIT) -> None:
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
        """
        acquire the lock
        """
        self._condition.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        release the lock
        """
        self._condition.release()

    @property
    def wait_for(self):
        """
        see `threading.Condition.wait_for`

        notes:
        - make sure to acquire the lock (via `with ...`) before calling
        """
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
        - True if the caller should run (start)
          - state is set to State.STARTING
          - the caller should set state to State.STARTED when done
        - False if the caller should return

        raises:
        - StateTransitionError if currently stopping or done
        """
        with self._condition:
            if self._state in (State.STARTING, State.STARTED):
                return False
            elif self._state is State.STOPPING:
                raise StateTransitionError("can't start: currently stopping")
            elif self._state is State.DONE:
                raise StateTransitionError("can't start: already done")
            self._state = State.STARTING
            return True

    def prepare_stop(
        self, *, init_behavior: InitStopBehavior = InitStopBehavior.ALLOW
    ) -> bool:
        """
        prepare to stop

        args:
        - init_behavior: behavior when in State.INIT
          - ALLOW: return True (see below)
          - NOP: return False
          - RAISE: raise StateTransitionError

        returns:
        - True if the caller should run (stop)
          - state is set to State.STOPPING
          - the caller should set state to State.STOPPED or State.DONE when done
        - False if the caller should return

        raises:
        - StateTransitionError if currently starting
        """
        with self._condition:
            if self._state in (State.STOPPING, State.STOPPED, State.DONE):
                return False
            elif self._state is State.STARTING:
                raise StateTransitionError("can't stop: currently starting")
            elif self._state is State.INIT:
                match init_behavior:
                    case InitStopBehavior.ALLOW:
                        pass
                    case InitStopBehavior.NOP:
                        return False
                    case InitStopBehavior.RAISE:
                        raise StateTransitionError("can't stop: not started")
            self._state = State.STOPPING
            return True
