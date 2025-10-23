import queue
import threading
import time
from collections import deque

import numpy as np
import torch
from openwakeword.model import Model
import rclpy
from rclpy.node import Node
from hri_msgs.msg import AudioAndDeviceInfo, WakeWord


class WakeWordDetectorNode(Node):
    def __init__(self):
        super().__init__("wake_word_detector")
       
        # Select device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # Audio buffer for continuous processing
        self.audio_queue = queue.Queue()

        # Sliding window parameters
        self.sample_rate = (
            16000  # Default sample rate, will be updated from audio stream
        )
        self.window_duration = 2.0  # 2 seconds window
        self.step_duration = 0.5  # 0.5 seconds step

        # Calculate sizes in samples
        self.window_size_samples = int(
            self.sample_rate * self.window_duration
        )  # 32000 samples for 2s
        self.step_size_samples = int(
            self.sample_rate * self.step_duration
        )  # 8000 samples for 0.5s

        # Sliding window buffer (circular buffer)
        # We need enough space for the window plus some extra for incoming data
        self.buffer_capacity = self.window_size_samples + self.step_size_samples * 2
        self.audio_buffer = deque(maxlen=self.buffer_capacity)

        # Window processing control
        self.last_process_time = 0.0

        # Processing control
        self.is_processing = True
        self.wake_word_detected = False

        # Wake word and wake word model
        self.wake_word = "hey_pipi"

        # .onnx model path based on wake word
        self.model_path = "/workspace/src/speech_recognition/weights_openwakeword/hey_pipi.onnx"

        # Initialize OpenWakeWord model
        try:
            self.get_logger().info("Loading OpenWakeWord model...")
            
            # Check if model file exists
            import os
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Check if model file is not empty
            if os.path.getsize(self.model_path) == 0:
                raise ValueError(f"Model file is empty: {self.model_path}")
            
            self.get_logger().info(f"Model file found: {self.model_path} ({os.path.getsize(self.model_path)} bytes)")
            
            # wakeword_model_paths expects a list of paths
            self.oww_model = Model(wakeword_model_paths=[self.model_path])
            self.get_logger().info("OpenWakeWord model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load OpenWakeWord model: {e}")
            self.oww_model = None

        # Subscriber
        self.audio_subscription = self.create_subscription(
            AudioAndDeviceInfo,
            "audio_and_device_info",
            self.audio_callback,
            10,
        )

        # Publisher
        self.wake_word_publisher = self.create_publisher(WakeWord, "wake_word", 10)

        # Start audio processing thread
        self.processing_thread = threading.Thread(target=self.audio_processing_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def audio_callback(self, msg: AudioAndDeviceInfo):
        """Callback function for audio stream subscription with sliding window"""
        try:
            # Log device information once on first message
            if not hasattr(self, "_device_logged"):
                self.get_logger().info(
                    f"First audio received from device: {msg.device_name} "
                    f"(ID: {msg.device_id}, Rate: {msg.device_samplerate} Hz)"
                )
                self._device_logged = True
            # Extract audio data from message
            audio_data = np.array(msg.audio, dtype=np.float32)

            # Update sample rate if different and recalculate window sizes
            if (
                hasattr(msg, "device_samplerate")
                and msg.device_samplerate != self.sample_rate
            ):
                self.sample_rate = msg.device_samplerate
                self.window_size_samples = int(self.sample_rate * self.window_duration)
                self.step_size_samples = int(self.sample_rate * self.step_duration)
                self.buffer_capacity = (
                    self.window_size_samples + self.step_size_samples * 2
                )
                # Recreate buffer with new capacity
                old_data = list(self.audio_buffer)
                self.audio_buffer = deque(maxlen=self.buffer_capacity)
                self.audio_buffer.extend(
                    old_data[-self.buffer_capacity :]
                )  # Keep recent data

            # Add new audio data to the sliding window buffer
            self.audio_buffer.extend(audio_data)

            # Check if we have enough data for processing and if it's time for next step
            current_time = time.time()
            if (
                len(self.audio_buffer) >= self.window_size_samples
                and current_time - self.last_process_time >= self.step_duration
            ):
                # Extract 2-second window from the end of buffer
                window_data = np.array(
                    list(self.audio_buffer)[-self.window_size_samples :]
                )

                # Add window to processing queue
                self.audio_queue.put(window_data)

                self.last_process_time = current_time

            # Start processing thread if not already started
            if not self.processing_thread.is_alive():
                self.processing_thread = threading.Thread(
                    target=self.audio_processing_thread
                )
                self.processing_thread.daemon = True
                self.processing_thread.start()

        except Exception as e:
            self.get_logger().error(f"Error in audio callback: {e}")

    def detect_wake_word(self, window_data):
        """Use OpenWakeWord for Alexa-like detection with CUDA optimization"""
        if self.oww_model is None:
            return False

        try:
            # Convert float32 data to int16 format required by OpenWakeWord
            audio_int16 = (window_data * 32767).astype(np.int16)

            # Get predictions from OpenWakeWord
            predictions = self.oww_model.predict(audio_int16)

            if predictions:
                # Extract the confidence score for the specific wake word
                if self.wake_word in predictions:
                    confidence_score = float(predictions[self.wake_word])
                    self.get_logger().info(f"Wake word '{self.wake_word}' confidence: {confidence_score:.6f}")

                    return confidence_score

        except Exception as e:
            self.get_logger().error(f"Error in OpenWakeWord detection: {e}")

        return False

    def audio_processing_thread(self):
        """Thread function for processing sliding windows"""
        while self.is_processing:
            try:
                self.get_logger().debug("Waiting for audio data...")
                # Get window data with timeout
                window_data = self.audio_queue.get(timeout=1.0)

                # Process the 2-second window
                wake_word_probability = self.detect_wake_word(window_data)

                if wake_word_probability:

                    # Publish wake word detection
                    msg = WakeWord()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.wake_word_probability = wake_word_probability

                    self.wake_word_publisher.publish(msg)

                    # Reset for next detection
                    self.wake_word_detected = False

                    # Pause briefly to avoid rapid re-detections
                    time.sleep(0.1)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error in audio processing thread: {e}")

    def destroy_node(self):
        """Clean shutdown of sliding window wake word detector"""
        self.get_logger().info("Shutting down sliding window wake word detector...")

        self.is_processing = False

        # Clear buffers
        try:
            self.audio_buffer.clear()
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
        except Exception as e:
            self.get_logger().debug(f"Error clearing buffers: {e}")

        # Wait for processing thread
        if hasattr(self, "processing_thread") and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    wake_word_detector = WakeWordDetectorNode()

    try:
        rclpy.spin(wake_word_detector)
    except KeyboardInterrupt:
        wake_word_detector.get_logger().info("Keyboard interrupt received")
    finally:
        wake_word_detector.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
