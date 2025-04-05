"""Base class for singleton pattern implementation."""

import threading
from abc import ABC
from typing import Generic, TypeVar

T = TypeVar("T")  # For the concrete singleton type


class Singleton(Generic[T], ABC):
    """
    Base class for singletons.

    Type parameter T should be the concrete singleton class itself.
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> T:
        """Get the singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
