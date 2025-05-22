import logging
import os
import signal
import sys
import time
import traceback
from collections import OrderedDict

from sound_monitor.audio.input import Input
from sound_monitor.audio.record import Record
from sound_monitor.audio.yamnet import Scores, YAMNet
from sound_monitor.config import Config
from sound_monitor.util.mail import mail

_config = Config.get()
_logger = logging.getLogger(__name__)

_input = Input.get()
_yamnet = YAMNet.get()


class App:
    def __init__(self) -> None:
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        _config.init()

    def run(self) -> None:
        _logger.info("starting")

        _input.start()
        _yamnet.start()

        _yamnet.register_callback("main", self._callback_yamnet)

        _logger.info("starting loop")
        self._loop()
        _logger.info("stopping loop")

    def stop(self) -> None:
        _logger.info("stopping")

        # TODO stop loop and recordings

        for e in (
            _yamnet,
            _input,
        ):
            try:
                e.stop()
            except Exception:
                _logger.exception(f"{e.__class__.__name__} stop failed")

        _logger.info("stopped")

    def _signal_handler(self, sig, frame):
        _logger.info(f"signal received: {signal.Signals(sig).name}")
        self.stop()
        sys.exit(0)

    def _callback_yamnet(self, data: Scores) -> None:
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

    def _loop(self):
        while True:
            if _input.error is not None:
                raise RuntimeError(_input.error)
            if _yamnet.error is not None:
                raise RuntimeError(_yamnet.error)

            time.sleep(1)


def main() -> None:
    try:
        app = App()
        app.run()
    except Exception as e:
        try:
            _logger.exception("main failed")
            mail(
                subject=f"sound_monitor: {e}",
                body=traceback.format_exc(),
                attachments=[_config.log_path],
            )
            app.stop()
            sys.exit(1)
        except Exception:
            os._exit(1)


if __name__ == "__main__":
    main()
