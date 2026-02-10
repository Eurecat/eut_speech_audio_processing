import queue
import threading
import time
from collections import deque

import numpy as np
import rclpy
import torch
from hri_msgs.msg import AudioAndDeviceInfo, WakeWord
from openwakeword.model import Model
from rclpy.node import Node


class WakeWordDetectorNode(Node):
    def __init__(self):
        super().__init__("wake_word_node")

        # Declare and get ROS2 parameters
        # For string arrays, we need to use proper parameter declaration
        self.declare_parameter("wake_word_model_names", ["hey_jana"])
        self.declare_parameter("window_duration", 2.0)
        self.declare_parameter("step_duration", 0.5)
        self.declare_parameter("log_interval", 1.0)

        # Get parameter values - fix string array retrieval
        # This is a list of ONNX model names (without .onnx extension)
        try:
            # Use get_parameter().value for string arrays instead of string_array_value
            wake_word_param = self.get_parameter("wake_word_model_names")
            self.wake_word_models = wake_word_param.value

            # Debug log the parameter retrieval
            self.get_logger().info(
                f"Retrieved wake_word_model_names parameter: {self.wake_word_models}"
            )
            self.get_logger().debug(f"Parameter type: {type(self.wake_word_models)}")

            # If parameter is empty or None, fallback to default
            if not self.wake_word_models or not isinstance(self.wake_word_models, list):
                self.get_logger().warning(
                    "wake_word_model_names parameter is empty or invalid, using default"
                )
                self.wake_word_models = ["hey_jana"]

        except Exception as e:
            self.get_logger().error(f"Error retrieving wake_word_model_names parameter: {e}")
            self.wake_word_models = ["hey_jana"]  # fallback

        self.window_duration = (
            self.get_parameter("window_duration").get_parameter_value().double_value
        )
        self.step_duration = self.get_parameter("step_duration").get_parameter_value().double_value
        self.log_time = self.get_parameter("log_interval").get_parameter_value().double_value

        self.get_logger().info(f"Using wake word models: {self.wake_word_models}")
        self.get_logger().info(f"Number of models loaded: {len(self.wake_word_models)}")

        # Select device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        light_green = "\033[38;5;82m"
        reset = "\033[0m"
        self.get_logger().info(
            f"{light_green}Using device on wake word detection: {self.device}{reset}"
        )

        # Audio buffer for continuous processing
        self.audio_queue = queue.Queue()

        # Sliding window parameters
        self.sample_rate = 16000  # Default sample rate, will be updated from audio stream

        # Calculate sizes in samples
        self.window_size_samples = int(
            self.sample_rate * self.window_duration
        )  # 32000 samples for 2s
        self.step_size_samples = int(self.sample_rate * self.step_duration)  # 8000 samples for 0.5s

        # Sliding window buffer (circular buffer)
        # We need enough space for the window plus some extra for incoming data
        self.buffer_capacity = self.window_size_samples + self.step_size_samples * 2
        self.audio_buffer = deque(maxlen=self.buffer_capacity)

        # Window processing control
        self.last_process_time = 0.0

        # Logging control
        self.last_confidence_log_time = 0.0

        # Processing control
        self.is_processing = True
        self.wake_word_detected = False

        # Wake word model paths based on the ONNX model names from parameters
        # Each model file should be located at: /workspace/src/speech_recognition/weights_openwakeword/{model_name}.onnx
        self.model_paths = []
        for model_name in self.wake_word_models:
            model_path = f"/workspace/src/speech_recognition/weights_openwakeword/{model_name}.onnx"
            self.model_paths.append(model_path)

        # Initialize single OpenWakeWord model instance with multiple ONNX models
        self.oww_model = None
        try:
            self.get_logger().info("Loading OpenWakeWord models...")

            # Check if all model files exist
            import os

            valid_model_paths = []
            for i, model_path in enumerate(self.model_paths):
                model_name = self.wake_word_models[i]

                if not os.path.exists(model_path):
                    self.get_logger().warning(f"Model file not found: {model_path}")
                    continue

                # Check if model file is not empty
                if os.path.getsize(model_path) == 0:
                    self.get_logger().warning(f"Model file is empty: {model_path}")
                    continue

                self.get_logger().info(
                    f"Model file found: {model_path} ({os.path.getsize(model_path)} bytes)"
                )
                valid_model_paths.append(model_path)

            if not valid_model_paths:
                raise ValueError("No valid model files found")

            # Load all models at once - OpenWakeWord Model can handle multiple ONNX files
            self.oww_model = Model(wakeword_model_paths=valid_model_paths)
            self.get_logger().info(
                f"\033[92mOpenWakeWord model loaded successfully with {len(valid_model_paths)} ONNX models\033[0m"
            )

        except Exception as e:
            self.get_logger().error(f"Failed to load OpenWakeWord models: {e}")
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
            if hasattr(msg, "device_samplerate") and msg.device_samplerate != self.sample_rate:
                self.sample_rate = msg.device_samplerate
                self.window_size_samples = int(self.sample_rate * self.window_duration)
                self.step_size_samples = int(self.sample_rate * self.step_duration)
                self.buffer_capacity = self.window_size_samples + self.step_size_samples * 2
                # Recreate buffer with new capacity
                old_data = list(self.audio_buffer)
                self.audio_buffer = deque(maxlen=self.buffer_capacity)
                self.audio_buffer.extend(old_data[-self.buffer_capacity :])  # Keep recent data

            # Add new audio data to the sliding window buffer
            self.audio_buffer.extend(audio_data)

            # Check if we have enough data for processing and if it's time for next step
            current_time = time.time()
            if (
                len(self.audio_buffer) >= self.window_size_samples
                and current_time - self.last_process_time >= self.step_duration
            ):
                # Extract 2-second window from the end of buffer
                window_data = np.array(list(self.audio_buffer)[-self.window_size_samples :])

                # Add window to processing queue
                self.audio_queue.put(window_data)

                self.last_process_time = current_time

            # Start processing thread if not already started
            if not self.processing_thread.is_alive():
                self.processing_thread = threading.Thread(target=self.audio_processing_thread)
                self.processing_thread.daemon = True
                self.processing_thread.start()

        except Exception as e:
            self.get_logger().error(f"Error in audio callback: {e}")

    def detect_wake_word(self, window_data):
        """Use OpenWakeWord for wake word detection with multiple models"""
        if not self.oww_model:
            return 0.0

        try:
            # Convert float32 data to int16 format required by OpenWakeWord
            audio_int16 = (window_data * 32767).astype(np.int16)

            max_confidence_score = 0.0
            winning_model = None
            all_scores = {}

            # Get predictions from all models
            try:
                # Get predictions from OpenWakeWord
                predictions = self.oww_model.predict(audio_int16)

                for model_name, confidence_score in predictions.items():
                    all_scores[model_name] = confidence_score

                    # Track the maximum confidence score and which model achieved it
                    if confidence_score > max_confidence_score:
                        max_confidence_score = confidence_score
                        winning_model = model_name

            except Exception as e:
                self.get_logger().error(f"Error processing models: {e}")

            # Log confidence scores periodically
            current_time = time.time()
            if current_time - self.last_confidence_log_time >= self.log_time and all_scores:
                scores_str = ", ".join(
                    [f"{name}: {score:.6f}" for name, score in all_scores.items()]
                )
                if winning_model:
                    self.get_logger().info(
                        f"Wake word confidences [{scores_str}] - MAX: {max_confidence_score:.6f} from '{winning_model}'"
                    )
                else:
                    self.get_logger().info(f"Wake word confidences [{scores_str}] - No detection")
                self.last_confidence_log_time = current_time

            return max_confidence_score

        except Exception as e:
            self.get_logger().error(f"Error in OpenWakeWord detection: {e}")
            return 0.0

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
                    msg.wake_word_probability = float(wake_word_probability)

                    self.wake_word_publisher.publish(msg)

                    # Reset for next detection
                    self.wake_word_detected = False

                    # Pause briefly to avoid rapid re-detections
                    time.sleep(0.1)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error in audio processing thread: {e}")
                # Add more detailed error information
                import traceback

                self.get_logger().error(f"Traceback: {traceback.format_exc()}")

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
