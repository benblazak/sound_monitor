import logging
import signal
import sys
import time
from collections import OrderedDict

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
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    _config.init()
    _input.start()
    _yamnet.start()

    def yamnet_callback(data: Scores) -> None:
        mean = data.mean
        top = 100
        threshold = 0.01
        mean = OrderedDict(
            {
                k: v
                for k, v in sorted(mean.items(), key=lambda x: x[1], reverse=True)[:top]
                if v >= threshold
            }
        )
        for k, v in reversed(mean.items()):
            print(f"{v:.2f} - {k}")
        print()

    _yamnet.register_callback("main", yamnet_callback)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
