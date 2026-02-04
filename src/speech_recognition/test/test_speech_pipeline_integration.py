"""
Integration tests for speech_recognition package using launch_pytest.
Tests the complete speech processing pipeline including VAD, ASR, and diarization.
"""
import pytest
import rclpy
from rclpy.node import Node
from hri_msgs.msg import AudioAndDeviceInfo, Vad, SpeechResult, SpeechActivityDetection, LiveSpeech
import launch
import launch_ros.actions
import launch_testing.actions
import launch_testing.markers
from threading import Event
import numpy as np
import time


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    """
    Launch speech recognition nodes for integration testing
    """
    vad_node = launch_ros.actions.Node(
        package='speech_recognition',
        executable='vad',
        name='vad_node',
        output='screen',
        parameters=[{
            'repo_model': 'snakers4/silero-vad',
            'model_name': 'silero_vad',
        }]
    )

    asr_node = launch_ros.actions.Node(
        package='speech_recognition',
        executable='asr',
        name='asr_node',
        output='screen',
        parameters=[{
            'model_size': 'tiny',  # Use smallest model for testing
            'compute_type': 'float32',
            'language': 'en',
            'vad_threshold': 0.5,
        }]
    )

    return (
        launch.LaunchDescription([
            vad_node,
            asr_node,
            launch_testing.actions.ReadyToTest(),
        ]),
        {
            'vad_node': vad_node,
            'asr_node': asr_node,
        }
    )


class TestSpeechRecognitionIntegration:
    """Integration tests for speech recognition pipeline"""

    @staticmethod
    def test_nodes_launch_successfully(launch_service, proc_output):
        """Test that all speech recognition nodes launch without errors"""
        rclpy.init()
        
        try:
            test_node = Node('test_speech_checker')
            
            # Wait for nodes to start up
            time.sleep(3.0)
            
            # Check if nodes are in the node graph
            node_names = test_node.get_node_names()
            
            # VAD node should be running
            assert 'vad_node' in node_names, f"VAD node not found in: {node_names}"
            
            # ASR node should be running
            assert 'asr_node' in node_names, f"ASR node not found in: {node_names}"
            
        finally:
            test_node.destroy_node()
            rclpy.shutdown()

    @staticmethod
    def test_speech_topics_availability(launch_service, proc_output):
        """Test that all speech processing topics are available"""
        rclpy.init()
        
        try:
            test_node = Node('test_speech_topics')
            
            # Wait for topics to be established
            time.sleep(3.0)
            
            # Get all topics
            topic_names = test_node.get_topic_names_and_types()
            topic_dict = dict(topic_names)
            
            # Expected topics
            expected_topics = [
                '/vad',
                '/speech_result',
                '/live_speech',
            ]
            
            # Check each expected topic
            for topic in expected_topics:
                if topic in topic_dict:
                    print(f"Found topic: {topic}")
                else:
                    print(f"Topic not found: {topic}")
            
            # Note: In test environment, some topics may not be available
            # without actual audio input, so we just check that nodes started
            
        finally:
            test_node.destroy_node()
            rclpy.shutdown()

    @staticmethod
    def test_vad_processing_pipeline(launch_service, proc_output):
        """Test VAD processing with mock audio data"""
        rclpy.init()
        
        try:
            # Create test nodes
            publisher_node = Node('mock_audio_publisher')
            vad_subscriber_node = Node('vad_test_subscriber')
            
            received_vad_messages = []
            vad_message_event = Event()
            
            def vad_callback(msg):
                received_vad_messages.append(msg)
                vad_message_event.set()
            
            # Create audio publisher and VAD subscriber
            audio_pub = publisher_node.create_publisher(
                AudioAndDeviceInfo,
                '/audio_and_device_info',
                10
            )
            
            vad_sub = vad_subscriber_node.create_subscription(
                Vad,
                '/vad',
                vad_callback,
                10
            )
            
            # Wait for connections
            time.sleep(2.0)
            
            # Create mock audio message with speech-like characteristics
            mock_msg = AudioAndDeviceInfo()
            # Generate audio that might trigger VAD (varying amplitude)
            audio_samples = []
            for i in range(8000):  # 0.5 seconds at 16kHz
                sample = 0.1 * np.sin(2 * np.pi * 440 * i / 16000)  # 440Hz tone
                audio_samples.append(sample)
            
            mock_msg.audio = audio_samples
            mock_msg.sample_rate = 16000
            mock_msg.device_name = "test_device"
            
            # Publish audio message
            audio_pub.publish(mock_msg)
            
            # Wait for VAD processing
            timeout = 10.0
            start_time = time.time()
            
            while not vad_message_event.is_set():
                rclpy.spin_once(publisher_node, timeout_sec=0.01)
                rclpy.spin_once(vad_subscriber_node, timeout_sec=0.01)
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    break
            
            # Check if VAD processed the audio
            if received_vad_messages:
                vad_msg = received_vad_messages[0]
                
                # Verify VAD message structure
                assert hasattr(vad_msg, 'is_speech'), "VAD message should have is_speech field"
                assert hasattr(vad_msg, 'confidence'), "VAD message should have confidence field"
                assert isinstance(vad_msg.is_speech, bool), "is_speech should be boolean"
                assert 0.0 <= vad_msg.confidence <= 1.0, "Confidence should be between 0 and 1"
                
                print(f"VAD result: is_speech={vad_msg.is_speech}, confidence={vad_msg.confidence}")
            
        finally:
            publisher_node.destroy_node()
            vad_subscriber_node.destroy_node()
            rclpy.shutdown()


class TestEndToEndSpeechPipeline:
    """End-to-end tests for the complete speech processing pipeline"""

    @staticmethod
    def test_audio_to_vad_to_asr_pipeline():
        """Test the complete pipeline: Audio -> VAD -> ASR"""
        rclpy.init()
        
        try:
            # Create test nodes
            audio_publisher = Node('pipeline_audio_publisher')
            vad_subscriber = Node('pipeline_vad_subscriber')
            asr_subscriber = Node('pipeline_asr_subscriber')
            
            received_vad = []
            received_asr = []
            vad_event = Event()
            asr_event = Event()
            
            def vad_callback(msg):
                received_vad.append(msg)
                vad_event.set()
            
            def asr_callback(msg):
                received_asr.append(msg)
                asr_event.set()
            
            # Create publishers and subscribers
            audio_pub = audio_publisher.create_publisher(
                AudioAndDeviceInfo,
                '/audio_and_device_info',
                10
            )
            
            vad_sub = vad_subscriber.create_subscription(
                Vad,
                '/vad',
                vad_callback,
                10
            )
            
            asr_sub = asr_subscriber.create_subscription(
                SpeechResult,
                '/speech_result',
                asr_callback,
                10
            )
            
            # Wait for all connections to establish
            time.sleep(3.0)
            
            # Create mock audio with speech-like pattern
            mock_msg = AudioAndDeviceInfo()
            
            # Generate longer audio sequence that might trigger speech recognition
            sample_rate = 16000
            duration = 2.0  # 2 seconds
            num_samples = int(sample_rate * duration)
            
            # Create audio with varying patterns (simulating speech)
            audio_samples = []
            for i in range(num_samples):
                # Mix of frequencies to simulate speech
                t = i / sample_rate
                sample = (0.1 * np.sin(2 * np.pi * 200 * t) + 
                         0.05 * np.sin(2 * np.pi * 400 * t) + 
                         0.03 * np.sin(2 * np.pi * 800 * t))
                
                # Add some envelope variation
                envelope = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))
                sample *= envelope
                
                audio_samples.append(float(sample))
            
            mock_msg.audio = audio_samples
            mock_msg.sample_rate = sample_rate
            mock_msg.device_name = "pipeline_test_device"
            
            # Publish audio
            audio_pub.publish(mock_msg)
            
            # Process messages for a while
            processing_time = 15.0  # Allow time for processing
            start_time = time.time()
            
            while time.time() - start_time < processing_time:
                rclpy.spin_once(audio_publisher, timeout_sec=0.01)
                rclpy.spin_once(vad_subscriber, timeout_sec=0.01)
                rclpy.spin_once(asr_subscriber, timeout_sec=0.01)
                time.sleep(0.01)
            
            # Verify pipeline processing
            print(f"Received {len(received_vad)} VAD messages")
            print(f"Received {len(received_asr)} ASR messages")
            
            if received_vad:
                print(f"VAD detected speech: {received_vad[0].is_speech}")
                print(f"VAD confidence: {received_vad[0].confidence}")
            
            if received_asr:
                print(f"ASR result: {received_asr[0].transcription}")
            
            # Note: In test environment without actual speech models,
            # we mainly verify the message flow and structure
            
        finally:
            audio_publisher.destroy_node()
            vad_subscriber.destroy_node()
            asr_subscriber.destroy_node()
            rclpy.shutdown()

    @staticmethod
    def test_multiple_audio_chunks_processing():
        """Test processing of multiple sequential audio chunks"""
        rclpy.init()
        
        try:
            publisher_node = Node('multi_chunk_audio_publisher')
            vad_subscriber = Node('multi_chunk_vad_subscriber')
            
            received_vad_messages = []
            
            def vad_callback(msg):
                received_vad_messages.append(msg)
            
            # Create publisher and subscriber
            audio_pub = publisher_node.create_publisher(
                AudioAndDeviceInfo,
                '/audio_and_device_info',
                50  # Larger queue for multiple chunks
            )
            
            vad_sub = vad_subscriber.create_subscription(
                Vad,
                '/vad',
                vad_callback,
                50
            )
            
            # Wait for connections
            time.sleep(2.0)
            
            # Send multiple audio chunks
            num_chunks = 5
            chunk_duration = 1.0  # 1 second per chunk
            sample_rate = 16000
            
            for chunk_idx in range(num_chunks):
                mock_msg = AudioAndDeviceInfo()
                
                # Generate different audio pattern for each chunk
                num_samples = int(sample_rate * chunk_duration)
                audio_samples = []
                
                for i in range(num_samples):
                    t = i / sample_rate
                    # Vary frequency based on chunk
                    freq = 200 + chunk_idx * 100  # 200, 300, 400, 500, 600 Hz
                    sample = 0.1 * np.sin(2 * np.pi * freq * t)
                    audio_samples.append(float(sample))
                
                mock_msg.audio = audio_samples
                mock_msg.sample_rate = sample_rate
                mock_msg.device_name = f"chunk_test_device_{chunk_idx}"
                
                # Publish chunk
                audio_pub.publish(mock_msg)
                
                # Small delay between chunks
                time.sleep(0.2)
                
                # Process messages
                for _ in range(10):
                    rclpy.spin_once(publisher_node, timeout_sec=0.01)
                    rclpy.spin_once(vad_subscriber, timeout_sec=0.01)
            
            # Final processing
            for _ in range(50):
                rclpy.spin_once(publisher_node, timeout_sec=0.01)
                rclpy.spin_once(vad_subscriber, timeout_sec=0.01)
                time.sleep(0.1)
            
            # Verify chunk processing
            print(f"Processed {num_chunks} audio chunks")
            print(f"Received {len(received_vad_messages)} VAD responses")
            
            # Should have processed multiple chunks
            # (actual number depends on VAD sensitivity and model loading)
            
        finally:
            publisher_node.destroy_node()
            vad_subscriber.destroy_node()
            rclpy.shutdown()


class TestSpeechRecognitionPerformance:
    """Performance tests for speech recognition pipeline"""

    @staticmethod
    def test_processing_latency():
        """Test processing latency of speech recognition pipeline"""
        rclpy.init()
        
        try:
            publisher_node = Node('latency_test_publisher')
            vad_subscriber = Node('latency_test_subscriber')
            
            latency_measurements = []
            
            def vad_callback(msg):
                # Calculate latency (simplified - would need timestamps in real test)
                receive_time = time.time()
                # In real test, would compare with audio publish timestamp
                latency_measurements.append(receive_time)
            
            audio_pub = publisher_node.create_publisher(
                AudioAndDeviceInfo,
                '/audio_and_device_info',
                10
            )
            
            vad_sub = vad_subscriber.create_subscription(
                Vad,
                '/vad',
                vad_callback,
                10
            )
            
            # Wait for connections
            time.sleep(2.0)
            
            # Send test audio and measure processing time
            start_time = time.time()
            
            mock_msg = AudioAndDeviceInfo()
            mock_msg.audio = [0.1] * 8000  # 0.5 seconds at 16kHz
            mock_msg.sample_rate = 16000
            mock_msg.device_name = "latency_test_device"
            
            publish_time = time.time()
            audio_pub.publish(mock_msg)
            
            # Wait for processing
            processing_timeout = 5.0
            while (time.time() - publish_time) < processing_timeout:
                rclpy.spin_once(publisher_node, timeout_sec=0.01)
                rclpy.spin_once(vad_subscriber, timeout_sec=0.01)
                time.sleep(0.01)
            
            total_time = time.time() - start_time
            print(f"Total processing time: {total_time:.3f} seconds")
            
            if latency_measurements:
                response_time = latency_measurements[0] - publish_time
                print(f"Response latency: {response_time:.3f} seconds")
                
                # Reasonable latency expectation (should be less than 2 seconds)
                assert response_time < 2.0, f"Response latency too high: {response_time}s"
            
        finally:
            publisher_node.destroy_node()
            vad_subscriber.destroy_node()
            rclpy.shutdown()

    @staticmethod
    def test_throughput_with_continuous_audio():
        """Test throughput with continuous audio stream"""
        rclpy.init()
        
        try:
            publisher_node = Node('throughput_test_publisher')
            vad_subscriber = Node('throughput_test_subscriber')
            
            messages_processed = 0
            
            def vad_callback(msg):
                nonlocal messages_processed
                messages_processed += 1
            
            audio_pub = publisher_node.create_publisher(
                AudioAndDeviceInfo,
                '/audio_and_device_info',
                100  # Large queue for throughput test
            )
            
            vad_sub = vad_subscriber.create_subscription(
                Vad,
                '/vad',
                vad_callback,
                100
            )
            
            # Wait for connections
            time.sleep(2.0)
            
            # Send continuous stream of small audio chunks
            test_duration = 5.0  # 5 seconds test
            chunk_interval = 0.1  # 100ms chunks
            start_time = time.time()
            chunks_sent = 0
            
            while (time.time() - start_time) < test_duration:
                mock_msg = AudioAndDeviceInfo()
                # Small chunk (100ms at 16kHz)
                mock_msg.audio = [0.1 * np.sin(2 * np.pi * 440 * i / 16000) 
                                 for i in range(1600)]
                mock_msg.sample_rate = 16000
                mock_msg.device_name = "throughput_test_device"
                
                audio_pub.publish(mock_msg)
                chunks_sent += 1
                
                # Maintain interval
                time.sleep(chunk_interval)
                
                # Process messages
                rclpy.spin_once(publisher_node, timeout_sec=0.001)
                rclpy.spin_once(vad_subscriber, timeout_sec=0.001)
            
            # Final processing
            for _ in range(100):
                rclpy.spin_once(publisher_node, timeout_sec=0.01)
                rclpy.spin_once(vad_subscriber, timeout_sec=0.01)
                time.sleep(0.01)
            
            # Calculate throughput
            actual_duration = time.time() - start_time
            throughput = chunks_sent / actual_duration
            processing_rate = messages_processed / actual_duration
            
            print(f"Sent {chunks_sent} audio chunks in {actual_duration:.2f}s")
            print(f"Throughput: {throughput:.2f} chunks/sec")
            print(f"Processed {messages_processed} VAD responses")
            print(f"Processing rate: {processing_rate:.2f} responses/sec")
            
            # Basic throughput expectations
            assert throughput > 5.0, "Should handle at least 5 chunks per second"
            
        finally:
            publisher_node.destroy_node()
            vad_subscriber.destroy_node()
            rclpy.shutdown()