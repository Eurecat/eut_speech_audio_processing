import threading
import time as T

import numpy as np
import rclpy
import sounddevice as sd
from rclpy.node import Node
from std_msgs.msg import Bool

from hri_msgs.msg import AudioAndDeviceInfo

DTYPE = "float32"
CHANNELS = 1
CHUNK = 512
DISCONNECTION_TIMEOUT = 3  # seconds
DISCONNECTION_CHECK_INTERVAL = 1  # seconds
TEST_STREAM_DURATION = 0.1  # seconds


class AudioCapturingNode(Node):
    def __init__(self):
        super().__init__("audio_capturing_node")

        # Publishers
        self.device_disconnected_pub = self.create_publisher(
            Bool, "device_disconnected", 10
        )
        self.audio_and_device_info_pub = self.create_publisher(
            AudioAndDeviceInfo, "audio_and_device_info", 10
        )

        # Subscriber
        self.device_disconnected_sub = self.create_subscription(
            Bool, "device_disconnected", self.device_disconnected_callback, 10
        )

        # Threading control for disconnection check
        self.disconnection_check_running = True
        self.disconnection_check_lock = threading.Lock()

        self.handling_disconnection = False

        # Initial device setup
        self.stream = None
        self.device = None
        self.device_index = None
        self.device_samplerate = None
        self.device_channels = None

        # Setup a working device
        self.setup_working_device()

        # Initialize last callback time
        self.last_callback_time = T.time()

        # Start disconnection check in a separate thread
        self.disconnection_thread = threading.Thread(
            target=self.disconnection_check_loop, daemon=True
        )
        self.disconnection_thread.start()

    def get_available_input_devices(self, devices):
        return [i for i, dev in enumerate(devices) if dev["max_input_channels"] > 0]

    def get_device_parameters(self, device_index):
        """Extract device parameters for a given device index."""
        self.device_index = int(device_index)  # Ensure it's an integer
        self.device = self.devices[self.device_index]
        self.device_samplerate = int(self.device["default_samplerate"])
        self.device_channels = min(self.device["max_input_channels"], CHANNELS)

    def test_device_connection(self, device_index):
        """Test if a device can be successfully opened and used."""
        try:
            self.get_device_parameters(device_index)

            # Try to create and start a test stream
            test_stream = sd.InputStream(
                device=device_index,
                samplerate=self.device_samplerate,
                channels=self.device_channels,
                dtype=DTYPE,
                blocksize=CHUNK,
                latency="low",
            )
            test_stream.start()
            T.sleep(TEST_STREAM_DURATION)
            test_stream.stop()
            test_stream.close()
            return True
        except Exception as e:
            self.get_logger().warn(f"Device {device_index} test failed: {e}")
            return False

    def create_audio_stream(self, device_index):
        """Create and start an audio input stream for the given device."""
        try:
            self.get_device_parameters(device_index)

            self.get_logger().info(
                f"Selected device: {self.device['name']}; Samplerate: {self.device_samplerate}; Channels: {self.device_channels}"
            )

            # Create the input stream
            self.stream = sd.InputStream(
                device=device_index,
                samplerate=self.device_samplerate,
                channels=self.device_channels,
                dtype=DTYPE,
                blocksize=CHUNK,
                callback=self.input_callback,
                latency="low",
            )
            self.stream.start()
            self.get_logger().info(
                f"Successfully started audio stream with {self.device['name']} and samplerate: {self.device_samplerate}."
            )

        except Exception as e:
            self.get_logger().error(f"Failed to start stream: {e}")
            raise

    def setup_working_device(self):
        """Find and setup a working audio device. Keep trying until successful."""
        while True:
            try:
                self.devices = sd.query_devices()
                self.available_devices = self.get_available_input_devices(self.devices)

                self.device_index = self.select_device(self.available_devices)
                self.create_audio_stream(self.device_index)
                break

            except KeyboardInterrupt:
                self.get_logger().info("Operation cancelled by user.")
                raise
            except Exception as e:
                self.get_logger().error(f"Failed to setup device: {e}")
                self.get_logger().info("Please try selecting a different device.")

    def select_device(self, available_devices):
        while True:
            self.get_logger().info("Available devices:")
            for i in available_devices:
                dev = self.devices[i]
                self.get_logger().info(
                    f"{i}: {dev['name']} ({dev['max_input_channels']} channels)"
                )

            while True:
                try:
                    idx = int(input("Select device index: "))
                    if idx in available_devices:
                        # Test if the selected device actually works
                        if self.test_device_connection(idx):
                            self.get_logger().info(f"Device {idx} tested successfully.")
                            return idx
                        else:
                            self.get_logger().error(
                                f"Device {idx} is not working properly. Please select another device."
                            )
                            # Refresh device list and break inner loop to show updated list
                            self.devices = sd.query_devices()
                            available_devices = self.get_available_input_devices(
                                self.devices
                            )
                            break
                    else:
                        self.get_logger().warn(
                            f"Invalid index. Choose from {available_devices}"
                        )
                except KeyboardInterrupt:
                    self.get_logger().info("Operation cancelled.")
                    raise
                except Exception:
                    self.get_logger().warn(
                        f"Invalid index. Choose from {available_devices}"
                    )

    # Callback for audio input stream. Whenever new audio data is available, this function is called.
    def input_callback(self, indata, frames, time, status):
        with self.disconnection_check_lock:
            self.last_callback_time = T.time()

        # For mono: indata[:, 0] gives all samples from channel 0
        # For multichannel: indata[:, CHANNEL] gives all samples from specific channel
        audio_data = indata[:, 0]  # Get all samples from first channel

        if status:
            self.get_logger().warn(f"Stream status: {status}")

        rms = float(np.sqrt(np.mean(indata ** 2)))

        if rms >= 0:
            # Create and publish the message
            audio_msg = AudioAndDeviceInfo()
            audio_msg.header.stamp = self.get_clock().now().to_msg()
            audio_msg.audio = audio_data
            audio_msg.device_name = self.device["name"]
            audio_msg.device_id = self.device_index
            audio_msg.device_samplerate = float(self.device_samplerate)

            self.audio_and_device_info_pub.publish(audio_msg)

            if rms == 0:
                self.get_logger().warn("RMS is zero, probably the device is muted")

    def disconnection_check_loop(self):
        """Continuously check for device disconnection in a separate thread."""
        while self.disconnection_check_running:
            self.check_disconnection()
            T.sleep(DISCONNECTION_CHECK_INTERVAL)

    def check_disconnection(self):
        with self.disconnection_check_lock:
            time_since_last_callback = T.time() - self.last_callback_time
            self.get_logger().debug(
                f"Time since last callback: {time_since_last_callback:.2f} seconds"
            )

        if (
            time_since_last_callback > DISCONNECTION_TIMEOUT
            and not self.handling_disconnection
        ):
            # Publish device status as True (disconnected)
            msg = Bool()
            msg.data = True
            self.device_disconnected_pub.publish(msg)

            self.handling_disconnection = True

            self.get_logger().error(
                "No callback for 5 seconds. Device may be disconnected."
            )

            # Stop the input stream safely
            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                    self.get_logger().info("Input stream stopped.")
                except Exception as e:
                    self.get_logger().error(f"Error stopping stream: {e}")
                finally:
                    self.stream = None

    def device_disconnected_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info("Received device disconnected message.")

            # Keep trying until we find a working device
            try:
                # Setup a new working device
                self.stream = self.setup_working_device()

                # Reset disconnection handling flag and update last callback time
                with self.disconnection_check_lock:
                    self.handling_disconnection = False
                    self.last_callback_time = T.time()

            except KeyboardInterrupt:
                self.get_logger().info("Device selection cancelled by user.")
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
