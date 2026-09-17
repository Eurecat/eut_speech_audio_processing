import os
import subprocess

import numpy as np
import rclpy
from hri_msgs.msg import AudioAndDeviceInfo
from rclpy.node import Node

from audio_stream_manager.utils.audio_to_mp3_utils import (
    convert_wav_to_mp3,
    save_to_wav,
)


class AudioToMp3(Node):
    def __init__(self):
        super().__init__("audio_to_mp3")

        # Declare parameters (values come from audio_params.yaml)
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("temp_wav", "recording_temp.wav")
        self.declare_parameter("output_mp3", "/workspace/src/audio_stream_manager/recording.mp3")
        self.declare_parameter("flush_interval", 30.0)
        self.declare_parameter("max_buffer_seconds", 30.0)
        self.declare_parameter("recording_mode", "cyclic")
        self.declare_parameter("recordings_dir", "/workspace/src/audio_stream_manager/recordings")
        self.declare_parameter("cyclic_num_files", 4)

        self.sample_rate = self.get_parameter("sample_rate").get_parameter_value().integer_value
        self.temp_wav = self.get_parameter("temp_wav").get_parameter_value().string_value
        self.output_mp3 = self.get_parameter("output_mp3").get_parameter_value().string_value
        self.flush_interval = self.get_parameter("flush_interval").get_parameter_value().double_value
        self.max_buffer_seconds = self.get_parameter("max_buffer_seconds").get_parameter_value().double_value
        self.recording_mode = self.get_parameter("recording_mode").get_parameter_value().string_value
        self.recordings_dir = self.get_parameter("recordings_dir").get_parameter_value().string_value
        self.cyclic_num_files = self.get_parameter("cyclic_num_files").get_parameter_value().integer_value
        self._cyclic_index = 0
        self._last_mp3_save = 0.0

        if self.recording_mode == "cyclic":
            os.makedirs(self.recordings_dir, exist_ok=True)
            self._cyclic_temp_wav = os.path.join(self.recordings_dir, "_batch_temp.wav")
            self.get_logger().info(
                f"Recording mode: cyclic, keeping last {self.cyclic_num_files} batches of "
                f"~{self.max_buffer_seconds:.0f}s each in {self.recordings_dir}"
            )
        else:
            self.get_logger().info(f"Recording mode: full, writing entire session to {self.output_mp3}")

        self.subscription = self.create_subscription(
            AudioAndDeviceInfo,
            "/audio_and_device_info",
            self.audio_callback,
            10,
        )
        self.audio_buffer = []  # list of numpy float32 arrays
        self._chunks_received = 0
        self.create_timer(10.0, self._status_callback)
        self.create_timer(self.flush_interval, self._flush_timer_callback)
        self.get_logger().info("AudioToMp3 node started, subscribing to /audio_and_device_info")

    def _buffer_duration_seconds(self) -> float:
        total_samples = sum(len(c) for c in self.audio_buffer)
        return total_samples / self.sample_rate if self.sample_rate else 0.0

    def audio_callback(self, msg):
        chunk = np.array(msg.audio, dtype=np.float32)
        if chunk.size == 0:
            return
        self.audio_buffer.append(chunk)
        self._chunks_received += 1

        if self._buffer_duration_seconds() >= self.max_buffer_seconds:
            self._flush_buffer_to_disk(convert_to_mp3=(self.recording_mode == "cyclic"))

    def _status_callback(self):
        duration_s = self._buffer_duration_seconds()
        target = self.recordings_dir if self.recording_mode == "cyclic" else self.output_mp3
        self.get_logger().info(
            f"Recording: {duration_s:.1f}s buffered ({self._chunks_received} chunks) → {target}"
        )

    def _flush_timer_callback(self):
        if self.audio_buffer:
            self._flush_buffer_to_disk(convert_to_mp3=True)

    def _flush_buffer_to_disk(self, convert_to_mp3: bool = False) -> None:
        if not self.audio_buffer:
            return

        duration_s = self._buffer_duration_seconds()
        buffer_to_write = list(self.audio_buffer)
        self.audio_buffer.clear()

        if self.recording_mode == "cyclic":
            self._flush_cyclic(buffer_to_write, duration_s)
            return

        write_ok = save_to_wav(
            buffer_to_write,
            self.temp_wav,
            self.sample_rate,
            logger=self.get_logger(),
            append=os.path.exists(self.temp_wav),
        )
        if write_ok:
            self.get_logger().info(
                f"Flushed {duration_s:.1f}s to disk at {self.temp_wav}"
            )
            if convert_to_mp3:
                try:
                    convert_wav_to_mp3(self.temp_wav, self.output_mp3, logger=self.get_logger())
                    self.get_logger().info(f"Updated MP3 on disk: {self.output_mp3}")
                except subprocess.CalledProcessError as e:
                    self.get_logger().error(f"ffmpeg failed during periodic save: {e}")

    def _flush_cyclic(self, buffer_to_write: list, duration_s: float) -> None:
        write_ok = save_to_wav(
            buffer_to_write,
            self._cyclic_temp_wav,
            self.sample_rate,
            logger=self.get_logger(),
            append=False,
        )
        if not write_ok:
            return

        batch_mp3 = os.path.join(self.recordings_dir, f"recording_{self._cyclic_index}.mp3")
        try:
            convert_wav_to_mp3(self._cyclic_temp_wav, batch_mp3, logger=self.get_logger())
            self.get_logger().info(f"Flushed {duration_s:.1f}s batch to {batch_mp3}")
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"ffmpeg failed during cyclic save: {e}")
            return

        self._cyclic_index = (self._cyclic_index + 1) % self.cyclic_num_files


def main(args=None):
    rclpy.init(args=args)
    node = AudioToMp3()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping recording...")
    finally:
        node._flush_buffer_to_disk(convert_to_mp3=True)

        if node.recording_mode == "cyclic":
            if os.path.exists(node._cyclic_temp_wav):
                os.remove(node._cyclic_temp_wav)
        else:
            if os.path.exists(node.temp_wav):
                try:
                    convert_wav_to_mp3(node.temp_wav, node.output_mp3, logger=node.get_logger())
                except subprocess.CalledProcessError as e:
                    node.get_logger().error(f"ffmpeg failed: {e}")
            if os.path.exists(node.temp_wav):
                os.remove(node.temp_wav)

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
