import logging

from sound_monitor.audio.audio import Audio
from sound_monitor.config import Config
from sound_monitor.util.mail import mail

_config = Config.get()
_logger = logging.getLogger(__name__)

_audio = Audio.get()


def main():
    _config.init()
