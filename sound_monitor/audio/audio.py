import logging

import sounddevice as sd

from sound_monitor.config import Config
from sound_monitor.util.types.singleton import Singleton

_config = Config.get()
_logger = logging.getLogger(__name__)


class Audio(Singleton["Audio"]):
    pass
