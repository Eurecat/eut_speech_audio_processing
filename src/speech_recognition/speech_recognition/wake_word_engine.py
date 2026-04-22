import os
import queue
import threading
import time
from collections import deque
from typing import Callable, List, Optional

import numpy as np
import torch
from openwakeword.model import Model


class WakeWordEngine:
    """Loads OpenWakeWord models and runs sliding-window inference.

    Communicates outward only via one callback:
      - on_wake_word_detected(probability: float): called from the processing
        thread whenever a window yields a non-zero confidence score.
        The ROS2 node stamps and publishes from this callback.

    This class has zero ROS2 imports and can be tested standalone.
    """

    DEFAULT_SAMPLE_RATE = 16000

    def __init__(
        self,
        *,
        wake_word_models: List[str],
        model_base_path: str,
        window_duration: float,
        step_duration: float,
        on_wake_word_detected: Callable[[float], None],
        logger,
    ):
        self._logger = logger
        self._on_wake_word_detected = on_wake_word_detected
        self.window_duration = window_duration
        self.step_duration = step_duration

        self.sample_rate = self.DEFAULT_SAMPLE_RATE
        self._update_window_sizes()

        self.audio_queue: queue.Queue = queue.Queue()
        self.audio_buffer: deque = deque(maxlen=self._buffer_capacity)

        self.last_process_time = 0.0
        self.last_confidence_log_time = 0.0
        self.is_processing = True

        self.oww_model: Optional[Model] = self._load_models(wake_word_models, model_base_path)

        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()

    # ------------------------------------------------------------------
    # Window size helpers
    # ------------------------------------------------------------------

    def _update_window_sizes(self) -> None:
        self.window_size_samples = int(self.sample_rate * self.window_duration)
        self.step_size_samples = int(self.sample_rate * self.step_duration)
        self._buffer_capacity = self.window_size_samples + self.step_size_samples * 2

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self, model_names: List[str], base_path: str) -> Optional[Model]:
        valid_paths = []
        for name in model_names:
            path = os.path.join(base_path, f"{name}.onnx")
            if not os.path.exists(path):
                self._logger.warn(f"Model file not found: {path}")
                continue
            if os.path.getsize(path) == 0:
                self._logger.warn(f"Model file is empty: {path}")
                continue
            self._logger.info(f"Model file found: {path} ({os.path.getsize(path)} bytes)")
            valid_paths.append(path)

        if not valid_paths:
            self._logger.error("No valid model files found. Wake word detection disabled.")
            return None

        try:
            self._logger.info("Loading OpenWakeWord models...")
            model = Model(wakeword_model_paths=valid_paths)
            self._logger.info(
                f"OpenWakeWord model loaded successfully with {len(valid_paths)} ONNX models."
            )
            return model
        except Exception as e:
            self._logger.error(f"Failed to load OpenWakeWord models: {e}")
            return None

    def push_audio(self, audio_data: np.ndarray, sample_rate: float) -> None:
        """Feed a new audio chunk into the sliding window buffer.

        Call this from the ROS2 audio subscription callback.
        """
        sr = int(sample_rate)
        if sr != self.sample_rate:
            self._logger.info(
                f"Sample rate changed from {self.sample_rate} to {sr}. Recalculating window sizes."
            )
            self.sample_rate = sr
            old_data = list(self.audio_buffer)
            self._update_window_sizes()
            self.audio_buffer = deque(maxlen=self._buffer_capacity)
            self.audio_buffer.extend(old_data[-self._buffer_capacity :])

        self.audio_buffer.extend(audio_data)

        current_time = time.time()
        ready = len(self.audio_buffer) >= self.window_size_samples
        step_elapsed = current_time - self.last_process_time >= self.step_duration

        if ready and step_elapsed:
            window = np.array(list(self.audio_buffer)[-self.window_size_samples :])
            self.audio_queue.put(window)
            self.last_process_time = current_time

        if not self._processing_thread.is_alive():
            self._logger.warn("Processing thread died — restarting.")
            self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self._processing_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the processing thread to stop and wait for it."""
        self.is_processing = False
        self.audio_buffer.clear()
        # Drain the queue so the thread unblocks
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        if self._processing_thread.is_alive():
            self._processing_thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _detect(self, window_data: np.ndarray) -> float:
        """Run inference on one window. Returns max confidence across all models."""
        if not self.oww_model:
            return 0.0

        try:
            audio_int16 = (window_data * 32767).astype(np.int16)
            predictions = self.oww_model.predict(audio_int16)

            max_score = 0.0
            winning_model = None
            for model_name, score in predictions.items():
                if score > max_score:
                    max_score = score
                    # Don't
                    winning_model = model_name

            return max_score, winning_model

        except Exception as e:
            self._logger.error(f"Error in OpenWakeWord detection: {e}")
            return 0.0, None

    # ------------------------------------------------------------------
    # Processing thread
    # ------------------------------------------------------------------

    def _processing_loop(self) -> None:
        while self.is_processing:
            try:
                self._logger.debug("Waiting for audio data...")
                window_data = self.audio_queue.get(timeout=1.0)
                probability, winning_model = self._detect(window_data)

                if probability:
                    self._on_wake_word_detected(probability, winning_model)
                    time.sleep(0.01)  # Brief pause to avoid rapid re-detections

            except queue.Empty:
                continue
            except Exception as e:
                import traceback

                self._logger.error(f"Error in processing thread: {e}")
                self._logger.error(f"Traceback: {traceback.format_exc()}")
