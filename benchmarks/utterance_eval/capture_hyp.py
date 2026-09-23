#!/usr/bin/env python3
"""Record what the live pipeline says about one audio file, for score.py.

Listens on the same ROS domain as a docker-compose_mp3.yaml run and writes one
JSON with every /speech_result (text + speaker) and every
/speech_activity_detection event (diarization speaker changes). Times are in
seconds of *audio*: the stamp of the first /audio_and_device_info chunk is t=0,
and the mp3 publisher sends chunks in real time.

Exits by itself once audio stopped arriving and nothing was published for
--idle seconds.

    python3 capture_hyp.py --out hyp/<audio_stem>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import rclpy
from hri_msgs.msg import AudioAndDeviceInfo, SpeechActivityDetection, SpeechResult
from rclpy.node import Node


def stamp_sec(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class Capture(Node):
    def __init__(self, idle: float):
        super().__init__("utterance_eval_capture")
        self.idle = idle
        self.t0: float | None = None
        self.device = ""
        self.last_event = time.monotonic()
        self.results: list[dict] = []
        self.activity: list[dict] = []
        self.create_subscription(AudioAndDeviceInfo, "/audio_and_device_info", self._on_audio, 50)
        self.create_subscription(SpeechResult, "/speech_result", self._on_result, 50)
        self.create_subscription(SpeechActivityDetection, "/speech_activity_detection", self._on_activity, 50)
        self.get_logger().info("waiting for /audio_and_device_info ...")

    def _rel(self, t: float) -> float:
        return round(t - self.t0, 3) if self.t0 is not None else -1.0

    def _on_audio(self, msg: AudioAndDeviceInfo) -> None:
        if self.t0 is None:
            self.t0 = stamp_sec(msg)
            self.device = msg.device_name
            self.get_logger().info(f"audio started: {self.device}")
        self.last_event = time.monotonic()

    def _on_result(self, msg: SpeechResult) -> None:
        # locale carries "audio_ms=<n>;rtf=<x>", transcript_confidence carries processing ms (asr.py)
        extra = dict(kv.split("=", 1) for kv in msg.locale.split(";") if "=" in kv)
        self.results.append({
            "t_pub": self._rel(stamp_sec(msg)),
            "text": msg.transcript,
            "speaker": msg.speaker_id,
            "speaker_conf": round(msg.speaker_id_confidence, 3),
            "language": msg.language_code,
            "audio_ms": int(float(extra.get("audio_ms", 0))),
            "proc_ms": int(msg.transcript_confidence),
        })
        self.last_event = time.monotonic()
        self.get_logger().info(f"[{self.results[-1]['t_pub']:7.2f}] {msg.speaker_id}: {msg.transcript}")

    def _on_activity(self, msg: SpeechActivityDetection) -> None:
        self.activity.append({"t": self._rel(stamp_sec(msg)), "speaker": msg.speaker_id, "active": msg.active})
        self.last_event = time.monotonic()

    def done(self) -> bool:
        return self.t0 is not None and time.monotonic() - self.last_event > self.idle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--idle", type=float, default=8.0)
    args = ap.parse_args()

    rclpy.init()
    node = Capture(args.idle)
    try:
        while rclpy.ok() and not node.done():
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    out = {
        "schema_version": 1,
        "device": node.device,
        "captured_unix": int(time.time()),
        "results": node.results,
        "activity": node.activity,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: {len(node.results)} transcripts, {len(node.activity)} activity events")
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
