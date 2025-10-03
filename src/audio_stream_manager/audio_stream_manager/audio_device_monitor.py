import rclpy
from rclpy.node import Node
import sounddevice as sd
import time as T
from audio_stream_manager_interfaces.srv import CheckAudioDevice


class AudioDeviceMonitor(Node):
    def __init__(self):
        super().__init__("audio_device_monitor")

        # Create the service
        self.srv = self.create_service(
            CheckAudioDevice, "check_audio_device", self.check_device_callback
        )

    def check_device_callback(self, request, response):
        """
        Check if the given device is connected and if audio callback was recent enough.
        """
        device_name = request.device_name
        last_callback_time = request.last_callback_time

        # Check if device exists
        try:
            devices = sd.query_devices()
            device_connected = any(dev["name"] == device_name for dev in devices)
        except Exception as e:
            self.get_logger().error(f"Error querying audio devices: {e}")
            device_connected = False

        # Check last callback time
        time_since_callback = T.time() - last_callback_time
        threshold = 5.0  # seconds
        self.get_logger().info(
            f"Device '{device_name}' connected: {device_connected}, "
            f"time since last callback: {time_since_callback:.2f}s"
        )

        device_ok = device_connected and (time_since_callback <= threshold)

        response.success = device_ok

        return response


def main(args=None):
    rclpy.init(args=args)
    node = AudioDeviceMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down monitor.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
