import logging
import subprocess

_logger = logging.getLogger(__name__)


def audio_length(path: str) -> float:
    try:
        result = subprocess.run(
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
        duration = float(result.stdout.strip())
        return duration
    except subprocess.CalledProcessError as e:
        _logger.error(f"ffprobe exited with code {e.returncode}\n" + e.stderr.indent(2))
        raise RuntimeError(f"ffprobe exited with code {e.returncode}")
    except ValueError as e:
        _logger.error(f"failed to convert output to float: {e}")
        raise RuntimeError(f"failed to convert output to float: {e}")
