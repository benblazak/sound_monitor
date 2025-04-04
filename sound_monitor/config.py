from datetime import datetime
import logging
from pathlib import Path
from importlib import resources
from dataclasses import dataclass
import sys
from typing import Optional
import sounddevice as sd
import sysconfig

from sound_monitor.util.types.singleton import Singleton


@dataclass
class Config(Singleton["Config"]):

    data_dir: Path = Path(__file__).parent.parent / ".app" / "data"

    @property
    def app_dir(self) -> Path:
        if sys.prefix != sysconfig.get_config_var("base_prefix"):
            # in a venv
            return Path(sys.prefix) / "share" / "sound_monitor"
        # in a system install
        return Path.home() / ".local" / "share" / "sound_monitor"

    @property
    def download_dir(self) -> Path:
        return self.app_dir / "downloads"

    @property
    def yamnet_dir(self) -> Path:
        return self.download_dir / "yamnet"

    @property
    def yamnet_model(self) -> Path:
        return self.yamnet_dir / "yamnet.tflite"

    @property
    def yamnet_class_map(self) -> Path:
        return self.yamnet_dir / "yamnet_class_map.csv"

    date_format: str = "%Y-%m-%d"
    datetime_format: str = "%Y-%m-%d_%H-%M-%S"

    @property
    def prefix(
        self,
        time: Optional[datetime] = None,
        format: Optional[str] = None,
    ) -> str:
        time = time or datetime.now()
        format = format or self.datetime_format
        return time.strftime(format)

    @property
    def uma8_device_id(self) -> int | None:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if (
                "micarray" in device["name"].lower()
                and device["max_input_channels"] == 8
            ):
                return i

    def init(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.app_dir.mkdir(parents=True, exist_ok=True)

        log_path = self.data_dir / f"{self.prefix()}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(log_path),
            ],
        )


if __name__ == "__main__":
    print(getattr(Config.get(), sys.argv[1]))
