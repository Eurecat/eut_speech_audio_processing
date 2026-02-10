"""
Minimal unit tests for DiarizationObserver - ESSENTIAL TESTS ONLY
"""

from unittest.mock import Mock

from speech_recognition.diarization import DiarizationObserver


class TestDiarizationObserver:
    """Essential diarization tests only"""

    def test_observer_initialization(self):
        """Test DiarizationObserver initialization"""
        mock_node = Mock()
        observer = DiarizationObserver(mock_node)

        assert observer.node == mock_node
        assert isinstance(observer.known_diart_speakers, set)
        assert isinstance(observer.diart_to_eut_mapping, dict)
        assert len(observer.known_diart_speakers) == 0
        assert len(observer.diart_to_eut_mapping) == 0

    def test_speaker_mapping_creation(self):
        """Test creation of speaker mappings"""
        mock_node = Mock()
        observer = DiarizationObserver(mock_node)

        diart_speaker_1 = "speaker_001"
        diart_speaker_2 = "speaker_002"

        observer.known_diart_speakers.add(diart_speaker_1)
        observer.known_diart_speakers.add(diart_speaker_2)

        observer.diart_to_eut_mapping[diart_speaker_1] = "eut_speaker_1"
        observer.diart_to_eut_mapping[diart_speaker_2] = "eut_speaker_2"

        assert len(observer.known_diart_speakers) == 2
        assert len(observer.diart_to_eut_mapping) == 2
        assert observer.diart_to_eut_mapping[diart_speaker_1] == "eut_speaker_1"
        assert observer.diart_to_eut_mapping[diart_speaker_2] == "eut_speaker_2"
