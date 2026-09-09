import base64
import codecs
import json
import queue
import socket
import threading
from typing import Optional, Tuple

import numpy as np
import rclpy
from hri_msgs.msg import AudioAndDeviceInfo
from rclpy.node import Node


class AndroidAudioBridge(Node):
    """Receive Android audio chunks over TCP (NDJSON) and publish AudioAndDeviceInfo."""

    def __init__(self) -> None:
        super().__init__("android_audio_bridge")

        self.declare_parameter("bind_host", "0.0.0.0")
        self.declare_parameter("bind_port", 17000)
        self.declare_parameter("topic_name", "audio_and_device_info")
        self.declare_parameter("default_device_name", "android_phone")
        self.declare_parameter("default_device_id", 250)
        self.declare_parameter("default_samplerate", 16000.0)
        self.declare_parameter("max_queue_size", 64)

        self._bind_host = self.get_parameter("bind_host").get_parameter_value().string_value
        self._bind_port = int(self.get_parameter("bind_port").get_parameter_value().integer_value)
        self._topic_name = self.get_parameter("topic_name").get_parameter_value().string_value
        self._default_device_name = (
            self.get_parameter("default_device_name").get_parameter_value().string_value
        )
        self._default_device_id = int(
            self.get_parameter("default_device_id").get_parameter_value().integer_value
        )
        self._default_samplerate = float(
            self.get_parameter("default_samplerate").get_parameter_value().double_value
        )
        self._max_queue_size = int(
            self.get_parameter("max_queue_size").get_parameter_value().integer_value
        )

        self._publisher = self.create_publisher(AudioAndDeviceInfo, self._topic_name, 10)
        self._audio_queue: queue.Queue[Tuple[np.ndarray, str, int, float]] = queue.Queue(
            maxsize=self._max_queue_size
        )
        self._stop_event = threading.Event()
        self._server_thread = threading.Thread(target=self._serve, daemon=True)

        self._dropped_chunks = 0
        self._published_chunks = 0
        self._last_client: Optional[str] = None

        self.create_timer(0.01, self._drain_queue)
        self._server_thread.start()

        self.get_logger().info(
            f"Android audio bridge listening on {self._bind_host}:{self._bind_port}, "
            f"publishing to topic '{self._topic_name}'"
        )

    def _parse_audio_payload(self, payload: dict) -> Optional[np.ndarray]:
        audio = payload.get("audio")
        if isinstance(audio, list):
            try:
                return np.asarray(audio, dtype=np.float32)
            except Exception as exc:
                self.get_logger().warn(f"Invalid 'audio' array payload: {exc}")
                return None

        b64_data = payload.get("audio_b64_f32le")
        if isinstance(b64_data, str):
            try:
                raw = base64.b64decode(b64_data)
                return np.frombuffer(raw, dtype=np.float32)
            except Exception as exc:
                self.get_logger().warn(f"Invalid 'audio_b64_f32le' payload: {exc}")
                return None

        return None

    def _enqueue_chunk(
        self,
        audio_chunk: np.ndarray,
        device_name: str,
        device_id: int,
        samplerate: float,
    ) -> None:
        try:
            self._audio_queue.put_nowait((audio_chunk, device_name, device_id, samplerate))
        except queue.Full:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
            self._audio_queue.put_nowait((audio_chunk, device_name, device_id, samplerate))
            self._dropped_chunks += 1

    def _handle_payload(self, payload: dict, client_label: str) -> None:
        msg_type = payload.get("type", "audio_chunk")

        if msg_type == "stream_start":
            stream_id = payload.get("stream_id", "n/a")
            self.get_logger().info(f"Android stream_start from {client_label} stream_id={stream_id}")
            return

        if msg_type == "stream_end":
            stream_id = payload.get("stream_id", "n/a")
            self.get_logger().info(f"Android stream_end from {client_label} stream_id={stream_id}")
            return

        if msg_type != "audio_chunk":
            self.get_logger().warn(f"Unknown Android payload type '{msg_type}'")
            return

        audio_chunk = self._parse_audio_payload(payload)
        if audio_chunk is None or audio_chunk.size == 0:
            return

        device_name = str(payload.get("device_name", self._default_device_name))
        device_id = int(payload.get("device_id", self._default_device_id))
        samplerate = float(payload.get("sample_rate", self._default_samplerate))
        self._enqueue_chunk(audio_chunk, device_name, device_id, samplerate)

    def _handle_client(self, conn: socket.socket, addr: Tuple[str, int]) -> None:
        client_label = f"{addr[0]}:{addr[1]}"
        self._last_client = client_label
        self.get_logger().info(f"Android audio client connected: {client_label}")

        with conn:
            conn.settimeout(1.0)
            try:
                decoder = codecs.getincrementaldecoder("utf-8")()
                pending = ""
                while not self._stop_event.is_set():
                    try:
                        raw = conn.recv(4096)
                    except socket.timeout:
                        continue

                    if not raw:
                        break

                    try:
                        text = decoder.decode(raw)
                    except UnicodeDecodeError:
                        self.get_logger().warn(
                            f"Non UTF-8 payload from {client_label}. "
                            "This port expects NDJSON (not gRPC binary). Closing client."
                        )
                        return

                    pending += text
                    while "\n" in pending:
                        line, pending = pending.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError as exc:
                            self.get_logger().warn(f"Invalid JSON from {client_label}: {exc}")
                            continue
                        if not isinstance(payload, dict):
                            self.get_logger().warn(
                                f"Ignoring non-object JSON payload from {client_label}"
                            )
                            continue
                        self._handle_payload(payload, client_label)
            except OSError as exc:
                self.get_logger().warn(f"Socket error from {client_label}: {exc}")

        self.get_logger().info(f"Android audio client disconnected: {client_label}")

    def _serve(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._bind_host, self._bind_port))
        server.listen(5)
        server.settimeout(1.0)

        try:
            while not self._stop_event.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop_event.is_set():
                        break
                    raise
                self._handle_client(conn, addr)
        finally:
            try:
                server.close()
            except OSError:
                pass

    def _drain_queue(self) -> None:
        published_this_tick = 0
        while published_this_tick < 8:
            try:
                audio_chunk, device_name, device_id, samplerate = self._audio_queue.get_nowait()
            except queue.Empty:
                break

            msg = AudioAndDeviceInfo()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.audio = audio_chunk.tolist()
            msg.device_name = device_name
            msg.device_id = device_id
            msg.device_samplerate = float(samplerate)
            self._publisher.publish(msg)

            self._published_chunks += 1
            published_this_tick += 1

        if self._published_chunks > 0 and self._published_chunks % 500 == 0:
            self.get_logger().info(
                f"Published chunks={self._published_chunks}, dropped={self._dropped_chunks}, "
                f"last_client={self._last_client}"
            )

    def destroy_node(self) -> bool:
        self._stop_event.set()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AndroidAudioBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Android audio bridge")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
