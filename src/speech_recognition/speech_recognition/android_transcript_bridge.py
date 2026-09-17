import json
import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import rclpy
from hri_msgs.msg import SpeechResult
from rclpy.node import Node


@dataclass
class ClientSession:
    socket_obj: socket.socket
    queue_obj: queue.Queue[bytes]
    thread: threading.Thread


class AndroidTranscriptBridge(Node):
    """Expose ROS2 SpeechResult as NDJSON over TCP for Android consumers."""

    def __init__(self) -> None:
        super().__init__("android_transcript_bridge")

        self.declare_parameter("bind_host", "0.0.0.0")
        self.declare_parameter("bind_port", 17001)
        self.declare_parameter("max_client_queue_size", 128)
        self.declare_parameter("topic_name", "speech_result")

        self._bind_host = self.get_parameter("bind_host").get_parameter_value().string_value
        self._bind_port = int(self.get_parameter("bind_port").get_parameter_value().integer_value)
        self._max_client_queue_size = int(
            self.get_parameter("max_client_queue_size").get_parameter_value().integer_value
        )
        self._topic_name = self.get_parameter("topic_name").get_parameter_value().string_value

        # IMPORTANT: do not use attribute name `_clients` because rclpy.Node
        # reserves it internally for ROS service clients.
        self._tcp_clients: Dict[str, ClientSession] = {}
        self._tcp_clients_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._server_thread = threading.Thread(target=self._serve, daemon=True)

        self._messages_sent = 0
        self._dropped_messages = 0

        self.create_subscription(SpeechResult, self._topic_name, self._speech_result_callback, 10)
        self._server_thread.start()

        self.get_logger().info(
            f"Android transcript bridge listening on {self._bind_host}:{self._bind_port}, "
            f"subscribed to '{self._topic_name}'"
        )

    def _speech_result_to_bytes(self, msg: SpeechResult) -> bytes:
        processing_ms = int(msg.transcript_confidence)
        audio_duration_ms = None
        realtime_factor = None

        if msg.locale:
            parts = [p.strip() for p in msg.locale.split(";") if p.strip()]
            kv = {}
            for part in parts:
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                kv[k.strip()] = v.strip()

            if "audio_ms" in kv:
                try:
                    audio_duration_ms = int(kv["audio_ms"])
                except ValueError:
                    audio_duration_ms = None
            if "rtf" in kv:
                try:
                    realtime_factor = float(kv["rtf"])
                except ValueError:
                    realtime_factor = None

        if processing_ms <= 0:
            now_ms = int(time.time() * 1000)
            stamp_ms = (int(msg.header.stamp.sec) * 1000) + (int(msg.header.stamp.nanosec) // 1_000_000)
            processing_ms = max(0, now_ms - stamp_ms)

        payload = {
            "type": "speech_result",
            "transcript": msg.transcript,
            "transcript_confidence": 0.0,
            "speaker_id": msg.speaker_id,
            "speaker_id_confidence": float(msg.speaker_id_confidence),
            "language_code": msg.language_code,
            "locale": msg.locale,
            "processing_ms": processing_ms,
            "audio_duration_ms": audio_duration_ms,
            "realtime_factor": realtime_factor,
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
        }
        return (json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8")

    def _speech_result_callback(self, msg: SpeechResult) -> None:
        encoded = self._speech_result_to_bytes(msg)

        with self._tcp_clients_lock:
            sessions = list(self._tcp_clients.values())

        for session in sessions:
            try:
                session.queue_obj.put_nowait(encoded)
            except queue.Full:
                try:
                    session.queue_obj.get_nowait()
                except queue.Empty:
                    pass
                session.queue_obj.put_nowait(encoded)
                self._dropped_messages += 1

    def _client_sender_loop(self, client_id: str, conn: socket.socket, out_q: queue.Queue[bytes]) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    payload = out_q.get(timeout=1.0)
                except queue.Empty:
                    continue
                conn.sendall(payload)
                self._messages_sent += 1
                if self._messages_sent % 250 == 0:
                    self.get_logger().info(
                        f"Transcript bridge sent={self._messages_sent}, "
                        f"dropped={self._dropped_messages}, clients={len(self._tcp_clients)}"
                    )
        except OSError as exc:
            self.get_logger().warn(f"Transcript bridge client {client_id} disconnected: {exc}")
        finally:
            with self._tcp_clients_lock:
                session = self._tcp_clients.pop(client_id, None)
            if session is not None:
                try:
                    session.socket_obj.close()
                except OSError:
                    pass
                self.get_logger().info(f"Transcript bridge client removed: {client_id}")

    def _add_client(self, conn: socket.socket, addr: Tuple[str, int]) -> None:
        client_id = f"{addr[0]}:{addr[1]}"
        out_q: queue.Queue[bytes] = queue.Queue(maxsize=self._max_client_queue_size)
        thread = threading.Thread(
            target=self._client_sender_loop,
            args=(client_id, conn, out_q),
            daemon=True,
        )
        with self._tcp_clients_lock:
            self._tcp_clients[client_id] = ClientSession(
                socket_obj=conn,
                queue_obj=out_q,
                thread=thread,
            )
        thread.start()
        self.get_logger().info(f"Transcript bridge client connected: {client_id}")

    def _serve(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._bind_host, self._bind_port))
        server.listen(8)
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
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._add_client(conn, addr)
        finally:
            try:
                server.close()
            except OSError:
                pass

    def destroy_node(self) -> bool:
        self._stop_event.set()
        with self._tcp_clients_lock:
            sessions = list(self._tcp_clients.values())
            self._tcp_clients.clear()
        for session in sessions:
            try:
                session.socket_obj.close()
            except OSError:
                pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AndroidTranscriptBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Android transcript bridge")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
