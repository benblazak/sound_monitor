import logging
import sys
import sysconfig
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

from sound_monitor.util.types.singleton import Singleton


class Config(Singleton["Config"]):

    data_dir: Path = Path(__file__).parent.parent / ".app"

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

    # speed of sound (meters/second)
    # - varies with temperature (~0.6 m/s per °C) (and not with altitude)
    # - default 343.0 m/s at 20°C (standard room temperature)
    # - for Tucson AZ (5-39°C), the speed varies by ~5.9% over the temperature
    #   range. this introduces timing differences of ~15µs over our max array
    #   distance (8.6cm) which is small enough to not significantly impact
    #   direction finding
    speed_of_sound: float = 343.0

    uma8_sample_rate: int = 48000
    uma8_sample_format: np.dtype = np.float32

    # 8 channels, but only 7 mics, so the last channel is empty
    uma8_output_channels: int = 7

    # microphone array geometry (meters)
    # - positions from minidsp.cfg in order of input channels
    # - channel 0 is center, 1-6 form a hexagon
    uma8_mic_positions: tuple[tuple[float, float, float], ...] = (
        (0.000, 0.000, 0.000),  # center (ch 0)
        (0.000, 0.043, 0.000),  # top (ch 1)
        (0.037, 0.021, 0.000),  # top right (ch 2)
        (0.037, -0.021, 0.000),  # bottom right (ch 3)
        (0.000, -0.043, 0.000),  # bottom (ch 4)
        (-0.037, -0.021, 0.000),  # bottom left (ch 5)
        (-0.037, 0.021, 0.000),  # top left (ch 6)
    )

    @property
    def uma8_device_id(self) -> int | None:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if (
                "micarray" in device["name"].lower()
                and device["max_input_channels"] == 8
            ):
                return i

    date_format: str = "%Y-%m-%d"
    datetime_format: str = "%Y-%m-%d_%H-%M-%S"

    def prefix(
        self,
        time: datetime | None = None,
        format: str | None = None,
    ) -> str:
        time = time or datetime.now()
        format = format or self.datetime_format
        return time.strftime(format)

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
