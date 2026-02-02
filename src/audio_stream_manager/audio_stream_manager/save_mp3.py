import os
import subprocess
import wave

import numpy as np
import rclpy
from hri_msgs.msg import AudioAndDeviceInfo
from rclpy.node import Node

# sudo apt-get install ffmpeg

SAMPLE_RATE = 16000  # <-- change to your real sample rate
TEMP_WAV = "recording_temp.wav"
OUTPUT_MP3 = "/workspace/src/audio_stream_manager/recording.mp3"


class AudioToMp3(Node):
    def __init__(self):
        super().__init__("audio_to_mp3")
        self.subscription = self.create_subscription(
            AudioAndDeviceInfo,
            "/audio_and_device_info",  # <-- your topic name
            self.audio_callback,
            10,
        )
        self.audio_buffer = []  # list of numpy float32 arrays
        self.get_logger().info("AudioToMp3 node started, subscribing to /audio")

    def audio_callback(self, msg):
        # msg.data is a list/array of float32 samples
        chunk = np.array(msg.audio, dtype=np.float32)
        self.audio_buffer.append(chunk)

    def save_to_wav(self, wav_path: str):
        if not self.audio_buffer:
            self.get_logger().warn("No audio received, not writing WAV")
            return False

        samples = np.concatenate(self.audio_buffer)

        # Convert [-1.0, 1.0] float32 → int16 PCM
        samples = np.clip(samples, -1.0, 1.0)
        samples_int16 = (samples * 32767.0).astype(np.int16)

        # Write WAV file
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)  # 1 = mono, change to 2 for stereo
            wf.setsampwidth(2)  # 2 bytes = 16 bits
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(samples_int16.tobytes())

        self.get_logger().info(f"WAV written to {wav_path}, {len(samples_int16)} samples")
        return True

    def convert_wav_to_mp3(self, wav_path: str, mp3_path: str):
        # Uses ffmpeg to convert WAV → MP3
        cmd = [
            "ffmpeg",
            "-y",  # -y = overwrite
            "-i",
            wav_path,
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "2",  # quality (2 is high quality VBR)
            mp3_path,
        ]
        self.get_logger().info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        self.get_logger().info(f"MP3 written to {mp3_path}")


def main(args=None):
    rclpy.init(args=args)
    node = AudioToMp3()

    try:
        # Spin until you Ctrl+C
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping recording...")
    finally:
        # On shutdown, save audio buffer to WAV then MP3
        wrote_wav = node.save_to_wav(TEMP_WAV)
        if wrote_wav:
            try:
                node.convert_wav_to_mp3(TEMP_WAV, OUTPUT_MP3)
            except subprocess.CalledProcessError as e:
                node.get_logger().error(f"ffmpeg failed: {e}")
        # Optional: clean up temp file
        if os.path.exists(TEMP_WAV):
            os.remove(TEMP_WAV)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
