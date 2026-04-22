from dataclasses import dataclass


@dataclass
class ActiveDevice:
    """Holds the state of the currently active audio input device."""

    device: dict
    name: str
    index: int
    samplerate: int
    channels: int
