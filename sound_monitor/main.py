import logging
import signal

import numpy as np

from sound_monitor.audio.input import Input
from sound_monitor.audio.yamnet import Scores, YAMNet
from sound_monitor.config import Config
from sound_monitor.util.mail import mail

_config = Config.get()
_logger = logging.getLogger(__name__)

_input = Input.get()
_yamnet = YAMNet.get()

# TODO mail runtime errors


def signal_handler(sig, frame):
    _yamnet.stop()
    _input.stop()


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    _config.init()
    _input.start()
    _yamnet.start()

    def yamnet_callback(yamnet: YAMNet) -> None:
        top = 100
        threshold = 0.01
        indices = np.argsort(yamnet.scores.data)[-top:][::-1]
        indices = [i for i in indices if yamnet.scores.data[i] >= threshold]
        indices.reverse()
        for i in indices:
            print(f"{yamnet.scores.data[i]:.2f} - {Scores.class_names[i]}")
        print()

    _yamnet.register_callback("main", yamnet_callback)
