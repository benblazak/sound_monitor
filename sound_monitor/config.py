import logging
import sys
import sysconfig
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import sounddevice as sd

from sound_monitor.util.types.singleton import Singleton


class Config(Singleton["Config"]):

    timezone: ZoneInfo = ZoneInfo("America/Phoenix")
    data_dir: Path = Path(__file__).parent.parent / ".app"
    log_path: Path | None = None  # see init

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

    audio_gain_factor: float = 100.0  # multiplier
    audio_bandpass_filter: tuple[float, float] = (400, 4000)  # hz
    audio_mono_channel: int = 0  # center - see uma8_mic_positions
    audio_stereo_channels: tuple[int, int] = (
        1,
        4,
    )  # L and R facing away from my house - see uma8_mic_positions
    audio_buffer_seconds: int = 10

    direction_azimuth_offset: int = -90
    """
    degrees to add to array azimuth for true north
    - array azimuth 0 is at +x (usb cord)
    - if usb points east, north is at -90, offset is +90
    - if usb points west, north is at +90, offset is -90
    - if usb points south, north is at +-180, offset is +=180
    """

    speed_of_sound: float = 343.0
    """
    speed of sound (meters/second)
    - varies with temperature (~0.6 m/s per °C) (and not with altitude)
    - default 343.0 m/s at 20°C (standard room temperature)
    - for Tucson AZ (5-39°C), the speed varies by ~5.9% over the temperature
      range. this introduces timing differences of ~15µs over our max array
      distance (8.6cm) which is small enough to not significantly impact
      direction finding
    """

    uma8_sample_rate: int = 48000
    """
    sample rate (hz)
    - supported: 11200, 16000, 32000, 44100, 48000
    """

    uma8_sample_format: np.dtype = np.float32

    uma8_output_channels: int = 7
    """
    uma8 raw mode: 8ch of audio (7ch coming from the mems mics + 1 ch from spare
    pdm input input) are available as raw audio (non processed)
    """

    uma8_mic_positions: tuple[tuple[float, float, float], ...] = (
        (0.000, 0.000, 0.000),
        (0.000, 0.043, 0.000),
        (0.037, 0.021, 0.000),
        (0.037, -0.021, 0.000),
        (0.000, -0.043, 0.000),
        (-0.037, -0.021, 0.000),
        (-0.037, 0.021, 0.000),
    )
    """
    microphone array geometry (meters)
    - positions from minidsp.cfg in order of input channels
    - channel 0 is center, 1-6 form a hexagon
    - usb cord is on the right

    channels
    - 0: center
    - 1: top
    - 2: top right
    - 3: bottom right
    - 4: bottom
    - 5: bottom left
    - 6: top left
    """

    @property
    def uma8_device_id(self) -> int | None:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if (
                "micarray" in device["name"].lower()
                and device["max_input_channels"] == 8
            ):
                return i
        return None

    @property
    def record_ffmpeg_command_partial(self) -> list[str]:
        """partial ffmpeg command: append output filename to the end"""
        format_map = {
            np.float32: "f32le",
            np.int16: "s16le",
            np.int32: "s32le",
            np.float64: "f64le",
        }
        return [
            "ffmpeg",
            *["-f", format_map[self.uma8_sample_format]],
            *["-ar", str(self.uma8_sample_rate)],
            *["-ac", "2"],  # 2 channels
            *["-i", "-"],  # read from stdin
            "-af",  # mems eq
            "equalizer=f=80:width_type=h:width=200:g=4,"  # bass
            "equalizer=f=2500:width_type=q:width=1:g=-2,"  # presence
            "equalizer=f=8000:width_type=h:width=2000:g=-3,"  # treble
            "volume=1.0",  # see audio_gain_factor
            *["-c:a", "libmp3lame", "-q:a", "2"],  # mp3, vbr quality 2
        ]

    date_format: str = "%Y-%m-%d"
    time_format: str = "%H-%M-%S"
    tz_format: str = "%Z"
    datetime_format: str = f"{date_format}--{time_format}--{tz_format}"

    def prefix(
        self,
        time: datetime | None = None,
        format: str | None = None,
    ) -> str:
        time = time or datetime.now(self.timezone)
        format = format or self.datetime_format
        return time.strftime(format)

    def init(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.app_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.data_dir / f"{self.prefix()}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt=self.datetime_format,
            handlers=[
                logging.FileHandler(self.log_path),
            ],
        )


if __name__ == "__main__":
    print(getattr(Config.get(), sys.argv[1]))
