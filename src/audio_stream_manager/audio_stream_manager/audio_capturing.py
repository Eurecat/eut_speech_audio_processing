import time as T

import numpy as np
import rclpy
import sounddevice as sd
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from audio_stream_manager_interfaces.srv import CheckAudioDevice

# Audio configuration
DTYPE = "float32"
CHANNELS = 1
CHUNK = 256


class AudioCapturingNode(Node):
    """
    Node that captures audio from a device and publishes it.
    It consults the audio monitor service to handle device disconnections
    and automatically switches to another available device.
    """

    def __init__(self):
        super().__init__("audio_capturing_node")

        # Publisher for audio data
        self.audio_pub = self.create_publisher(Float32MultiArray, "audio", 10)

        # Query all available input devices if max_input_channels > 0, meaning they can be used for audio input
        self.devices = sd.query_devices()
        self.available_devices = [
            i for i, dev in enumerate(self.devices) if dev["max_input_channels"] > 0
        ]

        if not self.available_devices:
            self.get_logger().error("No input devices available!")
            raise RuntimeError("No input devices found")

        # Define initial device as a ROS2 parameter
        self.declare_parameter("device_name", "Jabra SPEAK 510 USB: Audio (hw:1,0)")
        self.device_name = self.get_parameter("device_name").value
        self.get_logger().info(f"Looking for device: {self.device_name}")

        # Search for the device index by name
        self.device_index = next(
            (
                i
                for i, dev in enumerate(self.devices)
                if dev["name"] == self.device_name
            ),
            None,
        )

        if self.device_index is None:
            self.get_logger().error(f"Device not found: {self.device_name}")
            raise RuntimeError(f"Device not found: {self.device_name}")

        self.device_samplerate = int(
            self.devices[self.device_index]["default_samplerate"]
        )
        self.device_channels = min(
            self.devices[self.device_index]["max_input_channels"], CHANNELS
        )

        # Service client to check device status
        self.cli = self.create_client(CheckAudioDevice, "check_audio_device")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for audio device monitor service...")

        # Last audio callback timestamp
        self.last_callback_time = T.time()

        # Start the audio stream
        self.stream = None
        self.start_stream()

        # Timer to periodically check service and handle reconnection
        self.create_timer(1.0, self.check_device_service)

    def start_stream(self):
        """Start the audio input stream with error handling"""
        device = self.devices[self.device_index]
        self.device_samplerate = int(device["default_samplerate"])
        self.device_channels = min(device["max_input_channels"], CHANNELS)

        try:
            self.stream = sd.InputStream(
                device=self.device_index,
                samplerate=self.device_samplerate,
                channels=self.device_channels,
                dtype=DTYPE,
                blocksize=CHUNK,
                callback=self.input_callback,
                latency="low",
            )
            self.stream.start()
            self.get_logger().info(
                f"Started audio stream on {device['name']} with samplerate {self.device_samplerate}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to start audio stream: {e}")
            self.stream = None

    def stop_stream(self):
        """Stop the audio input stream safely"""
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                self.get_logger().warn(f"Error stopping stream: {e}")
            self.stream = None

    def input_callback(self, indata, frames, time, status):
        """Audio callback: publish audio and update last callback timestamp"""
        self.last_callback_time = T.time()
        if status:
            self.get_logger().warn(f"Stream status: {status}")

        audio_data = indata[:, 0]

        # Calculate RMS (Root Mean Square) for logging
        rms = np.sqrt(np.mean(audio_data**2))

        if rms == 0:
            self.get_logger().warn("Silent audio detected!")

        msg = Float32MultiArray()
        msg.data = audio_data.astype(np.float32).tolist()
        self.audio_pub.publish(msg)

    def check_device_service(self):
        """
        Calls the monitor service asynchronously to check device status.
        """
        req = CheckAudioDevice.Request()
        req.device_name = self.device_name
        req.last_callback_time = self.last_callback_time

        future = self.cli.call_async(req)
        future.add_done_callback(self.handle_service_response)

    def handle_service_response(self, future):
        """
        Called when the service response is ready.
        """
        try:
            result = future.result()
            device_ok = result.success
            if not device_ok:
                self.get_logger().error(
                    "Device disconnected or audio inactive. Switching device..."
                )
                self.stop_stream()
                self.select_new_device()
                self.start_stream()
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def select_new_device(self):
        """
        Selects a new available device different from current one.
        """
        for idx in self.available_devices:
            if idx != self.device_index:
                self.device_index = idx
                self.get_logger().info(
                    f"Switching to device: {self.devices[idx]['name']}"
                )
                self.device_name = self.devices[idx]["name"]
                return
        self.get_logger().error("No alternative devices available!")


def main(args=None):
    rclpy.init(args=args)
    node = AudioCapturingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down audio capturing node.")
    finally:
        node.stop_stream()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
