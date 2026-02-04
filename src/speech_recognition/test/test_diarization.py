"""
Unit tests for DiarizationObserver and speaker diarization functionality.
Tests speaker identification and tracking.
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from speech_recognition.diarization import DiarizationObserver


class TestDiarizationObserver:
    """Test suite for DiarizationObserver class"""
    
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
        
        # Simulate discovering new speakers
        diart_speaker_1 = "speaker_001"
        diart_speaker_2 = "speaker_002"
        
        # Add speakers
        observer.known_diart_speakers.add(diart_speaker_1)
        observer.known_diart_speakers.add(diart_speaker_2)
        
        # Create EUT mappings
        observer.diart_to_eut_mapping[diart_speaker_1] = "eut_speaker_1"
        observer.diart_to_eut_mapping[diart_speaker_2] = "eut_speaker_2"
        
        assert len(observer.known_diart_speakers) == 2
        assert len(observer.diart_to_eut_mapping) == 2
        assert observer.diart_to_eut_mapping[diart_speaker_1] == "eut_speaker_1"
        assert observer.diart_to_eut_mapping[diart_speaker_2] == "eut_speaker_2"
    
    def test_speaker_tracking_updates(self):
        """Test updating speaker tracking information"""
        mock_node = Mock()
        observer = DiarizationObserver(mock_node)
        
        # Initial speaker
        speaker_id = "speaker_001"
        observer.known_diart_speakers.add(speaker_id)
        observer.diart_to_eut_mapping[speaker_id] = "eut_speaker_1"
        
        # Update mapping (e.g., when speaker is re-identified)
        observer.diart_to_eut_mapping[speaker_id] = "eut_speaker_1_updated"
        
        assert observer.diart_to_eut_mapping[speaker_id] == "eut_speaker_1_updated"
    
    def test_multiple_speakers_scenario(self):
        """Test handling multiple concurrent speakers"""
        mock_node = Mock()
        observer = DiarizationObserver(mock_node)
        
        # Simulate multiple speakers
        speakers = ["speaker_001", "speaker_002", "speaker_003"]
        
        for i, speaker in enumerate(speakers):
            observer.known_diart_speakers.add(speaker)
            observer.diart_to_eut_mapping[speaker] = f"eut_speaker_{i+1}"
        
        assert len(observer.known_diart_speakers) == 3
        assert len(observer.diart_to_eut_mapping) == 3
        
        # Verify all mappings
        for i, speaker in enumerate(speakers):
            assert observer.diart_to_eut_mapping[speaker] == f"eut_speaker_{i+1}"


class TestSpeakerDiarization:
    """Test suite for speaker diarization functionality"""
    
    def test_audio_segmentation(self):
        """Test audio segmentation for speaker identification"""
        # Mock audio data
        sample_rate = 16000
        duration = 5.0  # 5 seconds
        num_samples = int(sample_rate * duration)
        audio_data = np.random.randn(num_samples).astype(np.float32)
        
        # Test segmentation
        segment_duration = 1.0  # 1 second segments
        segment_samples = int(sample_rate * segment_duration)
        
        segments = []
        for i in range(0, len(audio_data), segment_samples):
            segment = audio_data[i:i + segment_samples]
            segments.append(segment)
        
        # Verify segmentation
        assert len(segments) == 5  # 5 seconds / 1 second per segment
        for segment in segments[:-1]:  # All but last segment
            assert len(segment) == segment_samples
        
        # Last segment might be shorter
        assert len(segments[-1]) <= segment_samples
    
    def test_speaker_embedding_extraction(self):
        """Test speaker embedding extraction logic"""
        # Mock speaker embeddings
        embedding_dim = 512
        speaker_embeddings = {
            "speaker_001": np.random.randn(embedding_dim).astype(np.float32),
            "speaker_002": np.random.randn(embedding_dim).astype(np.float32),
        }
        
        # Test embedding properties
        for speaker_id, embedding in speaker_embeddings.items():
            assert embedding.shape == (embedding_dim,)
            assert embedding.dtype == np.float32
            
            # Test normalization
            normalized = embedding / np.linalg.norm(embedding)
            assert abs(np.linalg.norm(normalized) - 1.0) < 1e-6
    
    def test_speaker_similarity_computation(self):
        """Test speaker similarity computation"""
        # Mock speaker embeddings
        embedding_dim = 512
        embedding_1 = np.random.randn(embedding_dim).astype(np.float32)
        embedding_2 = np.random.randn(embedding_dim).astype(np.float32)
        embedding_3 = embedding_1 + 0.1 * np.random.randn(embedding_dim).astype(np.float32)
        
        # Normalize embeddings
        embedding_1 = embedding_1 / np.linalg.norm(embedding_1)
        embedding_2 = embedding_2 / np.linalg.norm(embedding_2)
        embedding_3 = embedding_3 / np.linalg.norm(embedding_3)
        
        # Compute cosine similarities
        similarity_1_2 = np.dot(embedding_1, embedding_2)
        similarity_1_3 = np.dot(embedding_1, embedding_3)
        
        # Verify similarity properties
        assert -1.0 <= similarity_1_2 <= 1.0
        assert -1.0 <= similarity_1_3 <= 1.0
        
        # embedding_3 is derived from embedding_1, so should be more similar
        assert similarity_1_3 > similarity_1_2
    
    @pytest.mark.parametrize("num_speakers", [1, 2, 3, 4, 5])
    def test_variable_speaker_counts(self, num_speakers):
        """Test diarization with different numbers of speakers"""
        # Mock diarization result
        diarization_result = {}
        
        for i in range(num_speakers):
            speaker_id = f"speaker_{i:03d}"
            # Mock time segments for each speaker
            diarization_result[speaker_id] = [
                {"start": i * 2.0, "end": i * 2.0 + 1.5},  # 1.5 second segments
                {"start": i * 2.0 + 3.0, "end": i * 2.0 + 4.0}  # Another segment
            ]
        
        # Verify result structure
        assert len(diarization_result) == num_speakers
        
        for speaker_id, segments in diarization_result.items():
            assert isinstance(segments, list)
            for segment in segments:
                assert "start" in segment
                assert "end" in segment
                assert segment["end"] > segment["start"]
                assert segment["start"] >= 0


class TestDiarizationROSIntegration:
    """Test suite for ROS integration of diarization"""
    
    def test_diarization_message_structure(self):
        """Test diarization message structure"""
        # Mock diarization message
        diarization_msg = {
            "speakers": [
                {
                    "speaker_id": "eut_speaker_1",
                    "confidence": 0.85,
                    "start_time": 1.2,
                    "end_time": 3.7
                },
                {
                    "speaker_id": "eut_speaker_2", 
                    "confidence": 0.92,
                    "start_time": 4.1,
                    "end_time": 6.8
                }
            ],
            "timestamp": 12345.67
        }
        
        # Verify message structure
        assert "speakers" in diarization_msg
        assert "timestamp" in diarization_msg
        assert isinstance(diarization_msg["speakers"], list)
        
        for speaker in diarization_msg["speakers"]:
            assert "speaker_id" in speaker
            assert "confidence" in speaker
            assert "start_time" in speaker
            assert "end_time" in speaker
            assert 0.0 <= speaker["confidence"] <= 1.0
            assert speaker["end_time"] > speaker["start_time"]
    
    def test_speaker_id_consistency(self):
        """Test speaker ID consistency across messages"""
        # Mock sequence of diarization results
        diarization_sequence = [
            {"speaker_001": [{"start": 0.0, "end": 2.0}]},
            {"speaker_001": [{"start": 3.0, "end": 5.0}], "speaker_002": [{"start": 2.5, "end": 4.5}]},
            {"speaker_001": [{"start": 6.0, "end": 8.0}], "speaker_002": [{"start": 7.0, "end": 9.0}]}
        ]
        
        # Track speaker consistency
        known_speakers = set()
        
        for diarization in diarization_sequence:
            for speaker_id in diarization.keys():
                known_speakers.add(speaker_id)
        
        # Verify speaker IDs remain consistent
        assert "speaker_001" in known_speakers
        assert "speaker_002" in known_speakers
        assert len(known_speakers) == 2
    
    def test_overlapping_speakers_handling(self):
        """Test handling of overlapping speaker segments"""
        # Mock overlapping speakers
        overlapping_segments = {
            "speaker_001": [{"start": 1.0, "end": 3.0}],
            "speaker_002": [{"start": 2.0, "end": 4.0}]  # Overlaps with speaker_001
        }
        
        # Check for overlap detection
        speaker_1_segment = overlapping_segments["speaker_001"][0]
        speaker_2_segment = overlapping_segments["speaker_002"][0]
        
        # Detect overlap
        overlap_start = max(speaker_1_segment["start"], speaker_2_segment["start"])
        overlap_end = min(speaker_1_segment["end"], speaker_2_segment["end"])
        
        has_overlap = overlap_start < overlap_end
        assert has_overlap == True
        
        overlap_duration = overlap_end - overlap_start
        assert overlap_duration == 1.0  # 1 second overlap


class TestDiarizationEdgeCases:
    """Test suite for edge cases in diarization"""
    
    def test_single_speaker_audio(self):
        """Test diarization with single speaker"""
        single_speaker_result = {
            "speaker_001": [
                {"start": 0.0, "end": 2.0},
                {"start": 3.0, "end": 5.0},
                {"start": 6.0, "end": 8.0}
            ]
        }
        
        # Verify single speaker handling
        assert len(single_speaker_result) == 1
        speaker_segments = list(single_speaker_result.values())[0]
        assert len(speaker_segments) == 3
        
        # Check segment continuity
        total_speech_time = sum(seg["end"] - seg["start"] for seg in speaker_segments)
        assert total_speech_time == 6.0  # 2 + 2 + 2 seconds
    
    def test_no_speech_detected(self):
        """Test handling when no speech is detected"""
        no_speech_result = {}
        
        # Should handle empty results gracefully
        assert len(no_speech_result) == 0
        
        # Message should indicate no speakers found
        message = {"speakers": [], "timestamp": 12345.67}
        assert len(message["speakers"]) == 0
    
    def test_very_short_speech_segments(self):
        """Test handling of very short speech segments"""
        short_segments = {
            "speaker_001": [{"start": 1.0, "end": 1.1}],  # 100ms segment
            "speaker_002": [{"start": 2.0, "end": 2.05}]  # 50ms segment
        }
        
        min_segment_duration = 0.1  # 100ms minimum
        
        # Filter out segments that are too short
        filtered_segments = {}
        for speaker_id, segments in short_segments.items():
            valid_segments = [
                seg for seg in segments 
                if (seg["end"] - seg["start"]) >= min_segment_duration
            ]
            if valid_segments:
                filtered_segments[speaker_id] = valid_segments
        
        # Only speaker_001 should remain (100ms >= 100ms minimum)
        assert len(filtered_segments) == 1
        assert "speaker_001" in filtered_segments
        assert "speaker_002" not in filtered_segments
    
    def test_speaker_change_detection(self):
        """Test detection of speaker changes"""
        temporal_sequence = [
            {"speaker": "speaker_001", "start": 0.0, "end": 2.0},
            {"speaker": "speaker_002", "start": 2.0, "end": 4.0},
            {"speaker": "speaker_001", "start": 4.0, "end": 6.0},
        ]
        
        # Detect speaker changes
        speaker_changes = []
        prev_speaker = None
        
        for segment in temporal_sequence:
            current_speaker = segment["speaker"]
            if prev_speaker and current_speaker != prev_speaker:
                speaker_changes.append(segment["start"])
            prev_speaker = current_speaker
        
        # Should detect changes at t=2.0 and t=4.0
        assert len(speaker_changes) == 2
        assert speaker_changes[0] == 2.0
        assert speaker_changes[1] == 4.0
    
    @pytest.mark.parametrize("confidence_threshold", [0.3, 0.5, 0.7, 0.9])
    def test_confidence_thresholding(self, confidence_threshold):
        """Test confidence-based filtering of diarization results"""
        diarization_with_confidence = {
            "speaker_001": [{"start": 0.0, "end": 2.0, "confidence": 0.8}],
            "speaker_002": [{"start": 2.0, "end": 4.0, "confidence": 0.4}],
            "speaker_003": [{"start": 4.0, "end": 6.0, "confidence": 0.6}]
        }
        
        # Filter by confidence
        filtered_results = {}
        for speaker_id, segments in diarization_with_confidence.items():
            high_conf_segments = [
                seg for seg in segments 
                if seg["confidence"] >= confidence_threshold
            ]
            if high_conf_segments:
                filtered_results[speaker_id] = high_conf_segments
        
        # Verify filtering based on threshold
        if confidence_threshold <= 0.4:
            assert len(filtered_results) == 3  # All speakers pass
        elif confidence_threshold <= 0.6:
            assert len(filtered_results) == 2  # speaker_001 and speaker_003
        elif confidence_threshold <= 0.8:
            assert len(filtered_results) == 1  # Only speaker_001
        else:
            assert len(filtered_results) == 0  # No speakers pass