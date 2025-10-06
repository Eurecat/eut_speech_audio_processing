from diart import SpeakerDiarization
from diart.sources import MicrophoneAudioSource
from diart.inference import StreamingInference
from diart.sinks import RTTMWriter, Observer
from diart.blocks.diarization import SpeakerDiarizationConfig
import torch
from typing import Union, Tuple
from pyannote.core import Annotation


class SpeakerStatusPrinter(Observer):
    """Custom observer that prints speaker status in real-time"""

    def init(self):
        super().init()
        self.known_speakers = set()
        self.last_status = {}

    def extractprediction(self, value):
        """Extract prediction annotation from the value"""
        if isinstance(value, tuple):
            return value[0]  # Assuming prediction is the first element
        elif isinstance(value, Annotation):
            return value
        else:
            return None

    def on_next(self, value: Union[Tuple, Annotation]):
        """Process new diarization result and print speaker status"""
        prediction = self._extract_prediction(value)
        if prediction is None:
            return
        current_status = {}
        # Extract current active speakers from the annotation
        for segment, track, speaker in prediction.itertracks(yield_label=True):
            # Add speaker to known speakers
            if speaker not in self.known_speakers:
                self.known_speakers.add(speaker)
            # Mark speaker as active (speaking)
            current_status[speaker] = 1
        # Mark all known speakers that are not currently active as inactive
        for speaker in self.known_speakers:
            if speaker not in current_status:
                current_status[speaker] = 0
        # Only print if status has changed from last time
        if current_status != self.last_status:
            print("-" * 50)
            # Print status for all known speakers
            for speaker in sorted(self.known_speakers):
                speaker_num = speaker.replace("speaker", "")
                print(f"Speaker {speaker_num}: {current_status[speaker]}")
            self.last_status = current_status.copy()

    def on_error(self, error: Exception):
        """Handle errors"""
        print(f"Error in speaker status monitoring: {error}")

    def on_completed(self):
        """Handle completion"""
        print("Speaker status monitoring completed")


# Force GPU usage if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Configure for better multi-speaker detection
config = SpeakerDiarizationConfig(
    #     device=device,
    tau_active=0.7,  # Lower threshold for voice activity detection
    #     rho_update=0.2,  # Lower threshold for speaker clustering (more sensitive)
    delta_new=0.83,  # Lower threshold for new speaker detection (easier to detect new speakers)
    #     max_speakers=10,  # Increase if you expect more than 10 speakers
    #     duration=3.0,  # Shorter chunks might help with speaker transitions
    #     step=0.25,  # Smaller steps for finer temporal resolution
)
pipeline = SpeakerDiarization(config=config)
mic = MicrophoneAudioSource(device=8)
inference = StreamingInference(pipeline, mic, do_plot=True)
# Create instances of both observers
speaker_status_printer = SpeakerStatusPrinter()
rttm_writer = RTTMWriter(
    mic.uri,
    "/home/coghri/dev/whisper-streaming/audio_manager/src/audio_manager/tests/output/file.rttm",
)
# Attach both observers
inference.attach_observers(speaker_status_printer, rttm_writer)
# Start the streaming inference
prediction = inference()
