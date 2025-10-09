import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import torch
import time
import queue
import audio_stream_manager_interfaces
from audio_stream_manager_interfaces.msg import Asr, AudioAndDeviceInfo, Vad


class ASRNode(Node):
    def __init__(self):
        super().__init__("asr_node")
        self.get_logger().info("ASR Node has been started.")

        # Initialize ASR components here

        # Subscibers
        self.audio_and_device_info_sub = self.create_subscription(
            AudioAndDeviceInfo,
            "audio_and_device_info",
            self.audio_and_device_info_callback,
            10,
        )
        self.vad_sub = self.create_subscription(
            Vad,
            "vad",
            self.vad_callback,
            10,
        )
        # self.speaking_sub = self.create_subscription(
        #     Bool,
        #     "speaking",
        #     self.speaking_callback,
        #     10,
        # )

        # Publishers
        self.asr_pub = self.create_publisher(
            Asr,
            "asr",
            10,
        )
        # self.speaking_pub = self.create_publisher(
        #     Bool,
        #     "speaking",
        #     10,
        # )

        self.vad_activated = False
        self.time_vad_on = 0.0
        self.time_vad_off = 0.0
        self.time_last_vad_on = 0.0

        self.audio_chunk = queue.Queue()

    def audio_and_device_info_callback(self, msg: AudioAndDeviceInfo):
        if self.vad_activated:
            self.audio_chunk.put(msg.audio)

    def vad_callback(self, msg: Vad):
        # Process VAD message
        if msg.vad_probability > 0.5:
            self.time_vad_on = time.time()
            self.vad_activated = True
            # self.speaking_pub.publish(Bool(data=True))
        else:
            self.time_vad_off = time.time()
            self.vad_activated = False

    def create_chunk_to_transcribe(self, msg: Bool):
        if self.vad_activated and self.time_last_vad_on > 1.0:
            # Create audio chunk from queue
            audio_data = []
            while not self.audio_chunk.empty():
                audio_data.extend(self.audio_chunk.get())
            # Transcribe audio chunk
