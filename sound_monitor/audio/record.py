import logging
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from typing import Self

from sound_monitor.audio.block import Block
from sound_monitor.audio.input import Input
from sound_monitor.audio.util import audio_trim
from sound_monitor.config import Config

_config = Config.get()
_logger = logging.getLogger(__name__)

_input = Input.get()


class Record:
    def __init__(self, name: str | None = None) -> None:
        self.name: str | None = name
        self.path: Path | None = None

        self.error: list[str] = []
        
        self.start_time: float | None = None  # stream time
        self.stop_time: float | None = None  # stream time
        self.start_clock: datetime | None = None  # wall clock time
        self.stop_clock: datetime | None = None  # wall clock time

        self._queue: Queue[Block] = Queue()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._last_block: Block | None = None
        self._start_event = threading.Event()  # Used to signal when recording has actually started

    def start(self, time: float | None = None) -> Self:
        # If already started or stopped, don't allow starting again
        if self._thread is not None:
            return self

        with _input.buffer_lock:
            # If time is specified, find matching block in the buffer
            if time:
                for block in _input.buffer:
                    if block.time <= time < block.time + _input.block_seconds:
                        self._queue.put(block)
                        break  # We only need the one matching block
                
                # If queue is empty, time wasn't found
                if self._queue.empty():
                    _logger.warning(
                        f"Recording start time {time} not found in buffer, will start with next block"
                    )
            
            # Register callback to receive new blocks
            _input.register_callback(f"record-{id(self)}", self._callback)

        # Start worker thread
        self._thread = threading.Thread(target=self._worker)
        self._thread.start()
        
        # Wait for recording to actually start or fail
        # This will block until either the first block is processed or an error occurs
        started = self._start_event.wait(timeout=10)  # Add reasonable timeout to prevent infinite waiting
        
        # Check for timeout
        if not started and not self.start_time:
            self.error.append("Timed out waiting for recording to start")
            _logger.error(self.error[-1])
            self.stop()  # Ensure cleanup
            return self
        
        # If we have errors but no start_time, the initialization failed
        if not self.start_time and self.error:
            _logger.error("Recording failed to start")
            self.stop()  # Ensure cleanup
        
        return self

    def stop(self, time: float | None = None) -> Self:
        # If never started or already stopped, do nothing
        if self._thread is None or not self._thread.is_alive():
            return self
        
        # Signal worker to stop
        self._queue.put(None)
        self._thread.join()
        
        # Now that recording is stopped, handle trimming if needed
        if time and self.start_time and self.path:
            # Check if time is after what we recorded
            if self.stop_time and time > self.stop_time:
                _logger.warning(f"Stop time {time} is after recording end {self.stop_time}, using recording end")
                time = self.stop_time
                
            # Trim audio if needed
            try:
                desired_length = time - self.start_time
                if desired_length > 0:
                    audio_trim(self.path, length=desired_length)
                    
                    # Update stop time to reflect trimmed audio
                    self.stop_time = time
                    if self.start_clock:
                        self.stop_clock = self.start_clock + timedelta(seconds=desired_length)
                else:
                    _logger.warning(f"Stop time {time} is before start time {self.start_time}, not trimming")
            except Exception as e:
                self.error.append(f"Failed to trim audio: {str(e)}")
                _logger.error(self.error[-1])
        
        return self

    def _callback(self, input: Input) -> None:
        self._queue.put(input.buffer[-1])

    def _worker(self) -> None:
        try:
            while True:
                block = self._queue.get()
                self._queue.task_done()
                
                # Handle stop signal
                if block is None:
                    _input.remove_callback(f"record-{id(self)}")
                    
                    # Clear queue
                    while not self._queue.empty():
                        self._queue.get()
                        self._queue.task_done()
                    
                    # Update stop time/clock with last block
                    if self._last_block:
                        self.stop_time = self._last_block.time + _input.block_seconds
                        self.stop_clock = self._last_block.clock + timedelta(
                            seconds=_input.block_seconds
                        )
                    
                    # Close ffmpeg process
                    if self._process:
                        if self._process.stdin:
                            self._process.stdin.close()
                        try:
                            ret = self._process.wait(timeout=5)
                            if ret:
                                stderr = self._process.stderr.read().decode().strip() if self._process.stderr else ""
                                self.error.append(f"ffmpeg exited with return code {ret}\n{stderr}")
                                _logger.error(self.error[-1])
                        except subprocess.TimeoutExpired:
                            self.error.append("ffmpeg failed to stop, sending SIGTERM")
                            _logger.error(self.error[-1])
                            self._process.terminate()
                            try:
                                self._process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                self.error.append("ffmpeg failed to stop, sending SIGKILL")
                                _logger.error(self.error[-1])
                                self._process.kill()
                    
                    return
                
                # If this is the first block, initialize the recording
                if self.start_time is None:
                    # Initialize ffmpeg first
                    try:
                        # Create output file
                        self.path = _config.data_dir / (
                            _config.prefix(block.clock)
                            + (f"_{self.name}" if self.name else "")
                            + ".mp3"
                        )
                        
                        # Start ffmpeg process
                        self._process = subprocess.Popen(
                            _config.record_ffmpeg_command_partial + [self.path],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                        )
                        
                        if not self._process or self._process.stdin is None:
                            self.error.append("ffmpeg: failed to start or open stdin")
                            _logger.error(self.error[-1])
                            self._start_event.set()  # Signal start() that we've failed
                            return
                        
                        # Now that process is successfully started, set start times
                        self.start_time = block.time
                        self.start_clock = block.clock
                        self._start_event.set()  # Signal start() that we've started successfully
                        
                    except Exception as e:
                        self.error.append(f"Failed to initialize recording: {str(e)}")
                        _logger.error(self.error[-1])
                        self._start_event.set()  # Signal start() that we've failed
                        return
                
                # Process the block
                self._last_block = block
                self._process.stdin.write(block.recording_data.tobytes())
                
        except Exception as e:
            # Catch any unexpected errors and log them
            self.error.append(f"Error in recording worker: {str(e)}")
            _logger.error(self.error[-1])
            
            # If we haven't started yet, signal start() that we've failed
            if not self.start_time:
                self._start_event.set()
