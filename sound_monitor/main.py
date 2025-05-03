import logging

from sound_monitor.audio.input import Input
from sound_monitor.audio.record import Record
from sound_monitor.config import Config
from sound_monitor.util.mail import mail

_config = Config.get()
_logger = logging.getLogger(__name__)

_input = Input.get()

# TODO mail runtime errors


def main():
    _config.init()
    _input.init()
