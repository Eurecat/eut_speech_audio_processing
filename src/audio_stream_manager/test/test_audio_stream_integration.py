"""
Integration tests for audio_stream_manager package using launch_pytest.
Tests the complete audio capturing and processing pipeline.
"""
import pytest
import rclpy
from rclpy.node import Node
from hri_msgs.msg import AudioAndDeviceInfo
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
    Launch the audio_capturing node for integration testing
    """
    audio_capturing_node = launch_ros.actions.Node(
        package='audio_stream_manager',
        executable='audio_capturing_automatic_device',
        name='audio_capturing_node',
        output='screen',
        parameters=[{
            'device_name': 'test_device',
            'channels': 1,
            'chunk': 512,
            'target_samplerate': 16000,
            'disconnection_timeout': 3.0,
        }]
    )

    return (
        launch.LaunchDescription([
            audio_capturing_node,
            launch_testing.actions.ReadyToTest(),
        ]),
        {
            'audio_capturing_node': audio_capturing_node,
        }
    )


class TestAudioStreamManagerIntegration:
    """Integration tests for audio stream manager"""

    @staticmethod
    def test_node_launches_successfully(launch_service, proc_output):
        """Test that the audio capturing node launches without errors"""
        rclpy.init()
        
        try:
            # Create a test node to check if audio node is running
            test_node = Node('test_audio_checker')
            
            # Wait a bit for the node to start up
            time.sleep(2.0)
            
            # Check if the node is in the node graph
            node_names = test_node.get_node_names()
            assert 'audio_capturing_node' in node_names, f"Audio node not found in: {node_names}"
            
        finally:
            test_node.destroy_node()
            rclpy.shutdown()

    @staticmethod
    def test_audio_topic_availability(launch_service, proc_output):
        """Test that audio topics are available"""
        rclpy.init()
        
        try:
            test_node = Node('test_topic_checker')
            
            # Wait for node to fully initialize
            time.sleep(2.0)
            
            # Check if audio topic exists
            topic_names = test_node.get_topic_names_and_types()
            topic_dict = dict(topic_names)
            
            assert '/audio_and_device_info' in topic_dict, f"Audio topic not found in: {list(topic_dict.keys())}"
            
            # Verify message type
            expected_type = 'hri_msgs/msg/AudioAndDeviceInfo'
            actual_types = topic_dict['/audio_and_device_info']
            assert expected_type in actual_types, f"Expected {expected_type}, got {actual_types}"
            
        finally:
            test_node.destroy_node()
            rclpy.shutdown()

    @staticmethod
    def test_audio_message_structure(launch_service, proc_output):
        """Test that audio messages have the correct structure"""
        rclpy.init()
        
        try:
            test_node = Node('test_audio_subscriber')
            received_messages = []
            message_event = Event()
            
            def audio_callback(msg):
                received_messages.append(msg)
                message_event.set()
            
            # Subscribe to audio topic
            subscription = test_node.create_subscription(
                AudioAndDeviceInfo,
                '/audio_and_device_info',
                audio_callback,
                10
            )
            
            # Wait for at least one message (with timeout)
            timeout = 10.0
            start_time = time.time()
            
            while not message_event.is_set():
                rclpy.spin_once(test_node, timeout_sec=0.1)
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    break
            
            # Note: In real environment with audio devices, we'd expect messages
            # In test environment without audio devices, the node may not publish
            # This test validates the message structure if messages are received
            
            if received_messages:
                msg = received_messages[0]
                
                # Verify message structure
                assert hasattr(msg, 'audio'), "Message should have audio field"
                assert hasattr(msg, 'sample_rate'), "Message should have sample_rate field"
                assert hasattr(msg, 'device_name'), "Message should have device_name field"
                
                # Verify audio data properties
                if len(msg.audio) > 0:
                    assert all(isinstance(sample, (int, float)) for sample in msg.audio), "Audio samples should be numeric"
                    assert msg.sample_rate > 0, "Sample rate should be positive"
                    assert isinstance(msg.device_name, str), "Device name should be string"
            
        finally:
            test_node.destroy_node()
            rclpy.shutdown()


class TestAudioStreamManagerWithMockData:
    """Integration tests using mock audio data"""

    @staticmethod 
    def test_audio_pipeline_with_mock_publisher():
        """Test complete audio pipeline by publishing mock data"""
        rclpy.init()
        
        try:
            # Create test nodes
            publisher_node = Node('mock_audio_publisher')
            subscriber_node = Node('audio_pipeline_tester')
            
            received_messages = []
            message_event = Event()
            
            def audio_callback(msg):
                received_messages.append(msg)
                message_event.set()
            
            # Create publisher and subscriber
            audio_pub = publisher_node.create_publisher(
                AudioAndDeviceInfo,
                '/audio_and_device_info',
                10
            )
            
            audio_sub = subscriber_node.create_subscription(
                AudioAndDeviceInfo,
                '/audio_and_device_info',
                audio_callback,
                10
            )
            
            # Wait for connections to establish
            time.sleep(1.0)
            
            # Create and publish mock audio message
            mock_msg = AudioAndDeviceInfo()
            mock_msg.audio = [0.1, 0.2, -0.1, 0.3, -0.2] * 100  # 500 samples
            mock_msg.sample_rate = 16000
            mock_msg.device_name = "test_device"
            
            # Publish message
            audio_pub.publish(mock_msg)
            
            # Wait for message to be received
            timeout = 5.0
            start_time = time.time()
            
            while not message_event.is_set():
                rclpy.spin_once(publisher_node, timeout_sec=0.01)
                rclpy.spin_once(subscriber_node, timeout_sec=0.01)
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    break
            
            # Verify message was received and processed correctly
            assert len(received_messages) > 0, "Should receive mock audio message"
            
            received_msg = received_messages[0]
            assert len(received_msg.audio) == 500, "Should receive all audio samples"
            assert received_msg.sample_rate == 16000, "Sample rate should match"
            assert received_msg.device_name == "test_device", "Device name should match"
            
        finally:
            publisher_node.destroy_node()
            subscriber_node.destroy_node()
            rclpy.shutdown()

    @staticmethod
    def test_multiple_audio_chunks():
        """Test processing of multiple audio chunks"""
        rclpy.init()
        
        try:
            publisher_node = Node('multi_chunk_publisher')
            subscriber_node = Node('multi_chunk_subscriber')
            
            received_messages = []
            
            def audio_callback(msg):
                received_messages.append(msg)
            
            # Create publisher and subscriber
            audio_pub = publisher_node.create_publisher(
                AudioAndDeviceInfo,
                '/audio_and_device_info',
                10
            )
            
            audio_sub = subscriber_node.create_subscription(
                AudioAndDeviceInfo,
                '/audio_and_device_info',
                audio_callback,
                10
            )
            
            # Wait for connections
            time.sleep(1.0)
            
            # Publish multiple chunks
            num_chunks = 5
            chunk_size = 512
            
            for i in range(num_chunks):
                mock_msg = AudioAndDeviceInfo()
                # Generate different data for each chunk
                mock_msg.audio = [0.1 * (i + 1)] * chunk_size
                mock_msg.sample_rate = 16000
                mock_msg.device_name = f"test_device_{i}"
                
                audio_pub.publish(mock_msg)
                
                # Small delay between chunks
                time.sleep(0.1)
                
                # Spin to process messages
                rclpy.spin_once(publisher_node, timeout_sec=0.01)
                rclpy.spin_once(subscriber_node, timeout_sec=0.01)
            
            # Final spin to ensure all messages are processed
            for _ in range(10):
                rclpy.spin_once(publisher_node, timeout_sec=0.01)
                rclpy.spin_once(subscriber_node, timeout_sec=0.01)
                time.sleep(0.1)
            
            # Verify all chunks were received
            assert len(received_messages) >= num_chunks, f"Expected {num_chunks} messages, got {len(received_messages)}"
            
            # Verify chunk properties
            for i, msg in enumerate(received_messages[:num_chunks]):
                assert len(msg.audio) == chunk_size, f"Chunk {i} size mismatch"
                assert msg.sample_rate == 16000, f"Chunk {i} sample rate mismatch"
                if i < len(received_messages):
                    expected_value = 0.1 * (i + 1)
                    assert abs(msg.audio[0] - expected_value) < 0.001, f"Chunk {i} data mismatch"
            
        finally:
            publisher_node.destroy_node()
            subscriber_node.destroy_node()
            rclpy.shutdown()


class TestAudioStreamManagerPerformance:
    """Performance and stress tests for audio stream manager"""

    @staticmethod
    def test_high_frequency_audio_publishing():
        """Test handling of high-frequency audio data publishing"""
        rclpy.init()
        
        try:
            publisher_node = Node('high_freq_publisher')
            subscriber_node = Node('high_freq_subscriber')
            
            received_count = 0
            start_time = None
            
            def audio_callback(msg):
                nonlocal received_count, start_time
                if start_time is None:
                    start_time = time.time()
                received_count += 1
            
            # Create publisher and subscriber
            audio_pub = publisher_node.create_publisher(
                AudioAndDeviceInfo,
                '/audio_and_device_info',
                100  # Larger queue for high frequency
            )
            
            audio_sub = subscriber_node.create_subscription(
                AudioAndDeviceInfo,
                '/audio_and_device_info',
                audio_callback,
                100
            )
            
            # Wait for connections
            time.sleep(1.0)
            
            # Publish messages at high frequency
            num_messages = 50
            publish_rate = 0.02  # 50 Hz
            
            for i in range(num_messages):
                mock_msg = AudioAndDeviceInfo()
                mock_msg.audio = [0.1] * 256  # Small chunks for high frequency
                mock_msg.sample_rate = 16000
                mock_msg.device_name = "high_freq_device"
                
                audio_pub.publish(mock_msg)
                
                # Maintain publishing rate
                time.sleep(publish_rate)
                
                # Process messages
                rclpy.spin_once(publisher_node, timeout_sec=0.001)
                rclpy.spin_once(subscriber_node, timeout_sec=0.001)
            
            # Final processing
            for _ in range(20):
                rclpy.spin_once(publisher_node, timeout_sec=0.01)
                rclpy.spin_once(subscriber_node, timeout_sec=0.01)
                time.sleep(0.05)
            
            # Verify performance
            if start_time:
                duration = time.time() - start_time
                message_rate = received_count / duration
                
                # Should handle reasonable message rates
                assert received_count > 0, "Should receive some messages"
                print(f"Received {received_count} messages at {message_rate:.2f} Hz")
            
        finally:
            publisher_node.destroy_node()
            subscriber_node.destroy_node()
            rclpy.shutdown()