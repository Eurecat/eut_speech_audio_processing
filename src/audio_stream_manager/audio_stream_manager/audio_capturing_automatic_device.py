import os
import sys
import threading
import time as T

import numpy as np
import rclpy
import sounddevice as sd
from rclpy.node import Node
from rclpy.parameter import Parameter

from hri_msgs.msg import AudioAndDeviceInfo


class suppress_stderr:
    """Context manager to silence ALSA/PortAudio C-level error prints."""

    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.saved_stderr = os.dup(2)
        os.dup2(self.null_fd, 2)

    def __exit__(self, *args):
        os.dup2(self.saved_stderr, 2)
        os.close(self.null_fd)
        os.close(self.saved_stderr)


class AudioCapturingNode(Node):
    def __init__(self):
        super().__init__("audio_capturing_node")

        # Declare ROS2 parameters
        self.declare_parameter("device_name", "jabra")
        self.declare_parameter("dtype", "float32")
        self.declare_parameter("channels", 1)
        self.declare_parameter("chunk", 512)
        self.declare_parameter("disconnection_timeout", 3.0)
        self.declare_parameter("disconnection_check_interval", 1.0)
        self.declare_parameter("test_stream_duration", 0.1)
        self.declare_parameter("primary_device_check_interval", 5.0)
        self.declare_parameter("target_samplerate", 16000)

        # Publishers
        self.audio_and_device_info_pub = self.create_publisher(
            AudioAndDeviceInfo, "audio_and_device_info", 10
        )

        # Internal device disconnection state
        self.device_disconnected = False

        # Threading control for disconnection check
        self.disconnection_check_running = True
        self.disconnection_check_lock = threading.Lock()

        self.handling_disconnection = False

        # Primary device monitoring
        self.primary_device_name = None  # Store the original device name
        self.primary_device_check_running = True
        self.primary_device_check_lock = threading.Lock()
        self.is_using_fallback_device = False

        # Initial device setup
        self.stream = None
        self.device = None
        self.device_name = None
        self.device_index = None
        self.device_samplerate = None
        self.device_channels = None
        self.device_muted = False

        # Buffer for maintaining consistent chunk size after resampling
        self.audio_buffer = np.array([], dtype=np.float32)
        self.target_chunk_size = 512

        # Setup a working device
        self.setup_working_device()

        # Initialize last callback time
        self.last_callback_time = T.time()

        # Start disconnection check in a separate thread
        self.disconnection_thread = threading.Thread(
            target=self.disconnection_check_loop, daemon=True
        )
        self.disconnection_thread.start()

        # Start primary device check in a separate thread
        self.primary_device_thread = threading.Thread(
            target=self.primary_device_check_loop, daemon=True
        )
        self.primary_device_thread.start()

    def get_available_input_devices(self, devices):
        return [i for i, dev in enumerate(devices) if dev["max_input_channels"] > 0]

    def find_devices_by_name(self, device_name):
        """Find devices whose names contain the specified device_name (case-insensitive)."""
        if not device_name:
            # If no device name specified, return all available devices
            return self.available_devices

        matching_devices = []
        device_name_upper = device_name.upper()

        for device_index in self.available_devices:
            device_name_from_system = self.devices[device_index]["name"].upper()
            if device_name_upper in device_name_from_system:
                matching_devices.append(device_index)
                self.get_logger().debug(
                    f"Found matching device: {self.devices[device_index]['name']} (index {device_index})"
                )

        return matching_devices

    def get_device_parameters(self, device_index):
        """Extract device parameters for a given device index."""
        device_index = int(device_index)  # Ensure it's an integer
        device = self.devices[device_index]
        device_samplerate = int(device["default_samplerate"])
        channels_param = (
            self.get_parameter("channels").get_parameter_value().integer_value
        )
        device_channels = min(device["max_input_channels"], channels_param)

        return device_index, device, device_samplerate, device_channels

    def test_device_connection(self, device_index):
        """Test if a device can be successfully opened and used, and is receiving audio (RMS > 0)."""
        try:
            device_index, device, device_samplerate, device_channels = (
                self.get_device_parameters(device_index)
            )

            # Get parameters
            dtype_param = self.get_parameter("dtype").get_parameter_value().string_value
            chunk_param = (
                self.get_parameter("chunk").get_parameter_value().integer_value
            )

            # Try to create and start a test stream
            with suppress_stderr():
                test_stream = sd.InputStream(
                    device=device_index,
                    samplerate=device_samplerate,
                    channels=device_channels,
                    dtype=dtype_param,
                    blocksize=chunk_param,
                    latency="low",
                )
                test_stream.start()

                # Record some audio data to check if device is receiving input
                audio_data, _ = test_stream.read(chunk_param)
                test_stream.stop()
                test_stream.close()

            # Check if device is receiving audio (RMS > 0)
            rms = float(np.sqrt(np.mean(audio_data**2)))
            if rms > 0:
                self.device = device
                self.get_logger().debug(
                    f"Device {device_index} ({self.device['name']}) is receiving audio (RMS: {rms:.6f})"
                )
                self.device_name = self.device["name"]
                self.device_index = device_index
                self.device_samplerate = device_samplerate
                self.device_channels = device_channels
                return True
            else:
                self.get_logger().warn(
                    f"Device {device_index} ({device['name']}) is not receiving audio (RMS: 0)"
                )
                return False

        except Exception as e:
            self.get_logger().debug(f"Device {device_index} test failed: {e}")
            return False

    def create_audio_stream(self, device_index):
        """Create and start an audio input stream for the given device."""
        try:
            device_index, device, device_samplerate, device_channels = (
                self.get_device_parameters(device_index)
            )

            self.get_logger().debug(
                f"Selected device: {device['name']}; Samplerate: {device_samplerate}; Channels: {device_channels}"
            )

            # Get parameters
            dtype_param = self.get_parameter("dtype").get_parameter_value().string_value
            chunk_param = (
                self.get_parameter("chunk").get_parameter_value().integer_value
            )

            # Create the input stream
            with suppress_stderr():
                self.stream = sd.InputStream(
                    device=device_index,
                    samplerate=device_samplerate,
                    channels=device_channels,
                    dtype=dtype_param,
                    blocksize=chunk_param,
                    callback=self.input_callback,
                    latency="low",
                )
                self.stream.start()
            self.get_logger().info(
                f"Successfully started audio stream with {device['name']} and samplerate: {device_samplerate}."
            )

        except Exception as e:
            self.get_logger().error(f"Failed to start stream: {e}")
            raise

    def setup_working_device(self):
        """Find and setup a working audio device automatically based on device_name parameter."""
        device_name_param = (
            self.get_parameter("device_name").get_parameter_value().string_value
        )

        # Store the primary device name when first setting up
        if self.primary_device_name is None and device_name_param != "":
            self.primary_device_name = device_name_param
            self.get_logger().info(
                f"Primary device name set to: {self.primary_device_name}"
            )
        self.devices = sd.query_devices()
        available_devices = self.get_available_input_devices(self.devices)

        # Print available devices for debugging
        self.get_logger().info("-" * 70)
        self.get_logger().info("Available devices:")
        for i in available_devices:
            dev = self.devices[i]
            self.get_logger().info(
                f"{i}: {dev['name']} ({dev['max_input_channels']} channels)"
            )
        self.get_logger().info("-" * 70)
        while True:
            try:
                self.devices = sd.query_devices()
                self.available_devices = self.get_available_input_devices(self.devices)

                # Check if this is a disconnection scenario (device_name is empty)
                if device_name_param == "":
                    # Mark that we're using a fallback device
                    with self.primary_device_check_lock:
                        self.is_using_fallback_device = True

                    # Try each available device in order until one receives messages
                    for device_index in self.available_devices:
                        device_name = self.devices[device_index]["name"]
                        self.get_logger().debug(
                            f"Testing device {device_index}: {device_name}"
                        )

                        if self.test_device_connection(device_index):
                            self.create_audio_stream(device_index)

                            # Clear audio buffer when connecting to new device
                            self.audio_buffer = np.array([], dtype=np.float32)

                            # Update the device_name parameter with the working device
                            self.set_parameters(
                                [
                                    Parameter(
                                        "device_name",
                                        Parameter.Type.STRING,
                                        device_name,
                                    )
                                ]
                            )

                            self.device_index = device_index
                            self.device_samplerate = self.devices[device_index][
                                "default_samplerate"
                            ]

                            self.get_logger().debug(
                                f"Successfully connected to device: {device_name}. Updated device_name parameter."
                            )
                            return  # Successfully connected to a device

                    # If no device worked in disconnection scenario
                    self.get_logger().error(
                        "No working audio devices found that are receiving input during disconnection recovery."
                    )

                else:
                    # Normal operation - find devices matching the parameter name
                    matching_devices = self.find_devices_by_name(device_name_param)

                    # Mark that we're using the primary device (or attempting to)
                    with self.primary_device_check_lock:
                        self.is_using_fallback_device = False

                    if not matching_devices:
                        if device_name_param:
                            self.get_logger().warn(
                                f"No devices found containing '{device_name_param}' in name. Trying all available devices."
                            )
                            matching_devices = self.available_devices
                        else:
                            matching_devices = self.available_devices

                    # Try each matching device until one works
                    for device_index in matching_devices:
                        self.get_logger().debug(
                            f"Testing device {device_index}: {self.devices[device_index]['name']}"
                        )
                        if self.test_device_connection(device_index):
                            self.create_audio_stream(device_index)

                            # Clear audio buffer when connecting to new device
                            self.audio_buffer = np.array([], dtype=np.float32)

                            return  # Successfully connected to a device

                    # If no device worked, wait and try again
                    self.get_logger().error(
                        "No working audio devices found that are receiving input. Retrying in 2 seconds..."
                    )

                T.sleep(2)

            except KeyboardInterrupt:
                self.get_logger().debug("Operation cancelled by user.")
                raise
            except Exception as e:
                self.get_logger().error(f"Failed to setup device: {e}")
                self.get_logger().debug("Retrying in 2 seconds...")
                T.sleep(2)

    # Callback for audio input stream. Whenever new audio data is available, this function is called.
    def input_callback(self, indata, frames, time, status):
        with self.disconnection_check_lock:
            self.last_callback_time = T.time()

        # For mono: indata[:, 0] gives all samples from channel 0
        # For multichannel: indata[:, CHANNEL] gives all samples from specific channel
        audio_data = indata[:, 0]  # Get all samples from first channel

        if status:
            self.get_logger().warn(f"Stream status: {status}")

        rms = float(np.sqrt(np.mean(indata**2)))

        if rms >= 0:
            target_samplerate = (
                self.get_parameter("target_samplerate")
                .get_parameter_value()
                .integer_value
            )

            if self.device_samplerate != target_samplerate:
                if (
                    not hasattr(self, "_last_resampled_device_index")
                    or self.device_index != self._last_resampled_device_index
                ):
                    self.get_logger().debug(
                        f"Resampling audio from {self.device_samplerate} to {target_samplerate}"
                    )
                    self._last_resampled_device_index = self.device_index
                audio_data = self.resample_audio(audio_data, target_samplerate)

            if rms == 0 and not self.device_muted:
                self.get_logger().warn("RMS is zero, probably the device is muted")
                self.device_muted = True
            elif rms > 0 and self.device_muted:
                self.get_logger().info(
                    "RMS is non-zero again, device has been re-activated"
                )
                self.device_muted = False

            # Add audio data to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])

            # Process chunks of target_chunk_size
            while len(self.audio_buffer) >= self.target_chunk_size:
                # Extract exactly target_chunk_size samples
                chunk_data = self.audio_buffer[: self.target_chunk_size]
                # Keep remaining samples in buffer
                self.audio_buffer = self.audio_buffer[self.target_chunk_size :]

                # Create and publish the message with consistent chunk size
                audio_msg = AudioAndDeviceInfo()
                audio_msg.header.stamp = self.get_clock().now().to_msg()
                audio_msg.audio = chunk_data
                audio_msg.device_name = self.device_name
                audio_msg.device_id = self.device_index
                audio_msg.device_samplerate = float(target_samplerate)

                self.audio_and_device_info_pub.publish(audio_msg)

    def resample_audio(self, audio_data, target_samplerate):
        """Resample audio data to the target samplerate."""
        try:
            import librosa

            resampled_audio = librosa.resample(
                audio_data,
                orig_sr=self.device_samplerate,
                target_sr=target_samplerate,
            )
            return resampled_audio
        except ImportError:
            self.get_logger().error(
                "librosa is not installed. Cannot resample audio. Please install librosa."
            )
            return audio_data

    def disconnection_check_loop(self):
        """Continuously check for device disconnection in a separate thread."""
        disconnection_check_interval = (
            self.get_parameter("disconnection_check_interval")
            .get_parameter_value()
            .double_value
        )
        while self.disconnection_check_running:
            self.check_disconnection()
            T.sleep(disconnection_check_interval)

    def primary_device_check_loop(self):
        """Continuously check if the primary device becomes available again."""
        primary_device_check_interval = (
            self.get_parameter("primary_device_check_interval")
            .get_parameter_value()
            .double_value
        )

        while self.primary_device_check_running:
            with self.primary_device_check_lock:
                should_check = (
                    self.is_using_fallback_device
                    and self.primary_device_name is not None
                    and self.primary_device_name != ""
                )

            if should_check:
                self.check_primary_device_recovery()

            T.sleep(primary_device_check_interval)

    def check_primary_device_recovery(self):
        """Check if the primary device is available and switch back to it if possible."""
        try:
            self.get_logger().debug(
                f"Checking if primary device '{self.primary_device_name}' is available..."
            )

            # Query current devices
            current_devices = sd.query_devices()
            current_available_devices = self.get_available_input_devices(
                current_devices
            )

            # Find devices matching the primary device name
            matching_devices = []
            primary_device_name_upper = self.primary_device_name.upper()

            for device_index in current_available_devices:
                device_name_from_system = current_devices[device_index]["name"].upper()
                if primary_device_name_upper in device_name_from_system:
                    matching_devices.append(device_index)

            if not matching_devices:
                self.get_logger().debug(
                    f"Primary device '{self.primary_device_name}' not found yet."
                )
                return

            # Test if the primary device is working
            for device_index in matching_devices:
                device_name = current_devices[device_index]["name"]
                self.get_logger().debug(
                    f"Testing primary device recovery: {device_name}"
                )

                if self.test_device_connection(device_index):
                    self.get_logger().debug(
                        f"Primary device '{device_name}' is available again! Switching back..."
                    )

                    # Stop current stream
                    if self.stream is not None:
                        try:
                            with suppress_stderr():
                                self.stream.stop()
                                self.stream.close()
                            self.get_logger().debug(
                                "Stopped current stream for primary device switch."
                            )
                        except Exception as e:
                            self.get_logger().error(
                                f"Error stopping current stream: {e}"
                            )
                        finally:
                            self.stream = None

                    # Update device_name parameter back to primary device
                    self.set_parameters(
                        [
                            Parameter(
                                "device_name",
                                Parameter.Type.STRING,
                                self.primary_device_name,
                            )
                        ]
                    )

                    # Create new stream with primary device
                    self.create_audio_stream(device_index)

                    # Clear audio buffer when switching devices
                    self.audio_buffer = np.array([], dtype=np.float32)

                    # Reset flags
                    with self.primary_device_check_lock:
                        self.is_using_fallback_device = False

                    with self.disconnection_check_lock:
                        self.last_callback_time = T.time()

                    self.get_logger().debug(
                        f"Successfully switched back to primary device: {device_name}"
                    )
                    return

        except Exception as e:
            self.get_logger().error(f"Error during primary device recovery check: {e}")

    def check_disconnection(self):
        with self.disconnection_check_lock:
            time_since_last_callback = T.time() - self.last_callback_time
            self.get_logger().debug(
                f"Time since last callback: {time_since_last_callback:.2f} seconds"
            )

        disconnection_timeout = (
            self.get_parameter("disconnection_timeout")
            .get_parameter_value()
            .double_value
        )

        if (
            time_since_last_callback > disconnection_timeout
            and not self.handling_disconnection
        ):
            # Set internal disconnection state
            self.device_disconnected = True

            self.set_parameters(
                [
                    Parameter(
                        "device_name",
                        Parameter.Type.STRING,
                        "",
                    )
                ]
            )

            self.handling_disconnection = True

            self.get_logger().error(
                "No callback for 3 seconds. Device may be disconnected."
            )

            # Stop the input stream safely
            if self.stream is not None:
                try:
                    with suppress_stderr():
                        self.stream.stop()
                        self.stream.close()
                    self.get_logger().debug("Input stream stopped.")
                except Exception as e:
                    self.get_logger().error(f"Error stopping stream: {e}")
                finally:
                    self.stream = None
                    self.audio_buffer = np.array([], dtype=np.float32)

            # Handle disconnection immediately
            self.handle_device_disconnection()

    def handle_device_disconnection(self):
        """Handle device disconnection internally without using topics"""
        if self.device_disconnected:
            self.get_logger().debug(
                "Handling device disconnection internally. Searching for any working device in order..."
            )

            # Keep trying until we find a working device
            try:
                # Setup a new working device automatically
                self.setup_working_device()

                # Reset disconnection handling flag and update last callback time
                with self.disconnection_check_lock:
                    self.handling_disconnection = False
                    self.last_callback_time = T.time()

                # Reset disconnection state
                self.device_disconnected = False

            except KeyboardInterrupt:
                self.get_logger().debug("Device selection cancelled by user.")
            except Exception as e:
                self.get_logger().error(f"Failed to setup new device: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = AudioCapturingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node.")
    finally:
        # Stop the disconnection check thread
        node.disconnection_check_running = False
        if (
            hasattr(node, "disconnection_thread")
            and node.disconnection_thread.is_alive()
        ):
            node.disconnection_thread.join(timeout=2.0)

        # Stop the primary device check thread
        node.primary_device_check_running = False
        if (
            hasattr(node, "primary_device_thread")
            and node.primary_device_thread.is_alive()
        ):
            node.primary_device_thread.join(timeout=2.0)

        # Stop the audio stream
        if node.stream is not None:
            try:
                node.stream.stop()
                node.stream.close()
            except Exception as e:
                node.get_logger().error(f"Error stopping stream during shutdown: {e}")

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
