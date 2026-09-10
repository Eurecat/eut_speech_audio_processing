#!/usr/bin/env python3

import os
import subprocess
import time

import numpy as np
import rclpy
from hri_msgs.msg import AudioAndDeviceInfo
from rclpy.node import Node


class AudioFilePublisher(Node):
    """Publish an audio file as live AudioAndDeviceInfo ROS2 chunks."""

    def __init__(self):
        super().__init__("audio_file_publisher")

        self.declare_parameter("input_file", "")
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("chunk_size", 512)
        self.declare_parameter("tail_silence", 2.0)
        self.declare_parameter("min_subscribers", 2)
        self.declare_parameter("start_delay", 3.0)

        self.input_file = self.get_parameter("input_file").value
        self.sample_rate = int(self.get_parameter("sample_rate").value)
        self.chunk_size = int(self.get_parameter("chunk_size").value)
        self.tail_silence = float(self.get_parameter("tail_silence").value)
        self.min_subscribers = int(self.get_parameter("min_subscribers").value)
        self.start_delay = float(self.get_parameter("start_delay").value)

        if not self.input_file:
            raise ValueError("input_file is required")
        if not os.path.isfile(self.input_file):
            raise FileNotFoundError(f"Audio file not found: {self.input_file}")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        self.pub = self.create_publisher(
            AudioAndDeviceInfo,
            "/audio_and_device_info",
            10,
        )

        self.get_logger().info(f"Decoding: {self.input_file}")
        self.audio = self._decode_with_ffmpeg(self.input_file)

        # Force ASR to close/process the final utterance.
        tail_samples = int(round(self.tail_silence * self.sample_rate))
        if tail_samples:
            self.audio = np.concatenate(
                (self.audio, np.zeros(tail_samples, dtype=np.float32))
            )

        self.index = 0
        self.started = False
        self.ready_since = None
        self.play_timer = None
        self.wait_timer = self.create_timer(0.5, self._wait_until_pipeline_ready)

        source_duration = max(
            0.0,
            (len(self.audio) - tail_samples) / float(self.sample_rate),
        )
        self.get_logger().info(
            f"Prepared {source_duration:.2f}s audio + {self.tail_silence:.2f}s tail silence; "
            f"{self.sample_rate} Hz mono float32, chunk={self.chunk_size}"
        )
        self.get_logger().info(
            f"Waiting for >= {self.min_subscribers} subscribers on /audio_and_device_info..."
        )

    def _decode_with_ffmpeg(self, path: str) -> np.ndarray:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            path,
            "-vn",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "pipe:1",
        ]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg is not installed in the image. "
                "Your existing audio_to_mp3 code already expects ffmpeg, "
                "so install it in the image if necessary."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed to decode {path}:\n{stderr}") from exc

        audio = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=True)
        if audio.size == 0:
            raise RuntimeError(f"Decoded file contains no samples: {path}")
        return audio

    def _wait_until_pipeline_ready(self):
        if self.started:
            return

        count = self.pub.get_subscription_count()

        if count < self.min_subscribers:
            self.ready_since = None
            self.get_logger().info(
                f"/audio_and_device_info subscribers: {count}/{self.min_subscribers}"
            )
            return

        now = time.monotonic()
        if self.ready_since is None:
            self.ready_since = now
            self.get_logger().info(
                f"Pipeline subscribers detected ({count}). "
                f"Waiting additional {self.start_delay:.1f}s before playback..."
            )
            return

        if now - self.ready_since < self.start_delay:
            return

        self.started = True
        self.wait_timer.cancel()

        period = self.chunk_size / float(self.sample_rate)
        self.get_logger().info(
            f"Starting MP3 playback -> /audio_and_device_info "
            f"({period * 1000.0:.1f} ms/chunk)"
        )
        self.play_timer = self.create_timer(period, self._publish_next_chunk)

    def _publish_next_chunk(self):
        if self.index >= len(self.audio):
            self.get_logger().info(
                "Finished audio file (including tail silence). Exiting MP3 source."
            )
            if self.play_timer is not None:
                self.play_timer.cancel()
            rclpy.shutdown()
            return

        end = min(self.index + self.chunk_size, len(self.audio))
        chunk = self.audio[self.index:end]

        if chunk.size < self.chunk_size:
            chunk = np.pad(chunk, (0, self.chunk_size - chunk.size))

        msg = AudioAndDeviceInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.audio = chunk.astype(np.float32, copy=False).tolist()
        msg.device_name = f"file:{os.path.basename(self.input_file)}"
        msg.device_id = -1
        msg.device_samplerate = float(self.sample_rate)

        self.pub.publish(msg)
        self.index = end


def main(args=None):
    rclpy.init(args=args)
    node = AudioFilePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()