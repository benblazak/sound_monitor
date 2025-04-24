import logging
import subprocess
from pathlib import Path

_logger = logging.getLogger(__name__)


def audio_length(path: str | Path) -> float:
    """
    find the length of an audio file using ffprobe
    """
    try:
        run = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        _logger.error(f"{e}\n" + e.stderr.indent(2))
        raise

    duration = float(run.stdout.strip())

    return duration


def audio_trim(
    path: str | Path,
    start: float | None = None,
    stop: float | None = None,
    length: float | None = None,
) -> None:
    """
    trim an audio file using ffmpeg

    args
    - start: -ss [position in seconds]
    - stop: -to [position in seconds]
    - length: -t [duration in seconds]

    notes
    - if start, stop, and length are None, returns without doing anything
    - if stop and length are not None, raises a ValueError
    """

    if start is None and stop is None and length is None:
        return
    if stop is not None and length is not None:
        raise ValueError("expecting only one of: stop, length")

    path = Path(path)
    trim_path = path.with_suffix(".tmp" + path.suffix)

    try:
        subprocess.run(
            [
                "ffmpeg",
                *["-i", str(path)],
                *(["-ss", start] if start else []),
                *(["-to", stop] if stop else []),
                *(["-t", length] if stop else []),
                *["-c", "copy"],  # stream copy to avoid re-encoding
                str(trim_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        _logger.error(f"{e}\n" + e.stderr.indent(2))
        trim_path.unlink(missing_ok=True)
        raise

    trim_path.rename(path)
