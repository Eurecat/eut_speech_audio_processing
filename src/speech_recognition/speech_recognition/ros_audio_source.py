from diart.sources import AudioSource
import numpy as np
import queue
import threading
from typing import Optional, Text


######################################################################################
################################# NEW AUDIO SOURCE ###################################
######################################################################################


class ROSAudioSource(AudioSource):
    """Custom audio source that reads from ROS2 topics instead of hardware.

    This source implements the AudioSource interface from diart, using reactive
    programming patterns (Subject) to stream audio data received from ROS2 topics.
    """

    def __init__(
        self, sample_rate: int, block_duration: float = 0.5, uri: Optional[Text] = None
    ):
        """
        Initialize ROS audio source.

        Parameters
        ----------
        sample_rate : int
            Sample rate of the audio in Hz
        block_duration : float, optional
            Duration of each audio block in seconds. Default is 0.5s
        uri : Text, optional
            Unique identifier for this source. If None, auto-generated
        """
        # Initialize parent class
        if uri is None:
            uri = f"ros://audio_topic@{sample_rate}Hz"
        super().__init__(uri=uri, sample_rate=sample_rate)

        self.block_duration = block_duration
        self.block_size = int(sample_rate * block_duration)

        # Thread-safe queue to buffer incoming audio chunks from ROS
        self.audio_queue = queue.Queue(maxsize=100)

        # Buffer to accumulate audio until we have a full block
        self.audio_buffer = np.array([], dtype=np.float32)

        # Control flags
        self._is_running = False
        self._read_thread = None
        self._lock = threading.Lock()

    @property
    def duration(self) -> Optional[float]:
        """The duration of the stream. Unknown for live ROS streams."""
        return None

    def add_audio_chunk(self, audio_data: np.ndarray):
        """
        Add audio data received from ROS topic.

        This method should be called from the ROS callback when new audio
        data arrives.

        Parameters
        ----------
        audio_data : np.ndarray
            Audio samples as a numpy array (float32)
        """
        if not self._is_running:
            return

        # Ensure audio data is float32 and 1D
        audio_data = np.asarray(audio_data, dtype=np.float32)
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()

        try:
            # Put audio in queue (non-blocking with timeout)
            self.audio_queue.put(audio_data, timeout=0.01)
        except queue.Full:
            # Drop oldest chunk if queue is full to prevent memory buildup
            # This ensures we stay close to real-time
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put(audio_data, timeout=0.01)
            except (queue.Empty, queue.Full):
                pass

    def _read_loop(self):
        """
        Internal loop that reads from the queue and emits blocks through the stream.

        This runs in a separate thread and continuously processes audio chunks,
        accumulating them into blocks of the correct size before emitting.
        """
        while self._is_running:
            try:
                # Try to get audio chunk from queue
                chunk = self.audio_queue.get(timeout=0.1)

                # Accumulate in buffer
                self.audio_buffer = np.concatenate([self.audio_buffer, chunk])

                # Emit complete blocks through the stream
                while len(self.audio_buffer) >= self.block_size:
                    block = self.audio_buffer[: self.block_size]
                    self.audio_buffer = self.audio_buffer[self.block_size :]

                    # CORRECCIÓN: Reshape to (1, samples) as expected by diart
                    block = block.reshape(1, -1)  # CAMBIO: De (-1, 1) a (1, -1)

                    # Emit through the reactive stream
                    self.stream.on_next(block)

            except queue.Empty:
                # No data available, continue waiting
                continue
            except Exception as e:
                # Log error and notify stream
                self.stream.on_error(e)
                break

        # Notify stream completion when stopping
        self.stream.on_completed()

    def read(self):
        """
        Start reading the source and yielding samples through the stream.

        This starts a background thread that continuously processes audio
        chunks from the ROS topic and emits them through the reactive stream.
        """
        with self._lock:
            if self._is_running:
                return  # Already running

            self._is_running = True

            # Start the read loop in a separate thread
            self._read_thread = threading.Thread(
                target=self._read_loop, daemon=True, name="ROSAudioSource-ReadThread"
            )
            self._read_thread.start()

    def close(self):
        """
        Stop reading the source and close all open streams.

        This cleanly shuts down the background thread and clears all buffers.
        """
        with self._lock:
            if not self._is_running:
                return  # Already stopped

            self._is_running = False

        # Wait for read thread to finish
        if self._read_thread is not None and self._read_thread.is_alive():
            self._read_thread.join(timeout=2.0)

        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Clear the buffer
        self.audio_buffer = np.array([], dtype=np.float32)

    def __enter__(self):
        """Context manager entry."""
        self.read()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
