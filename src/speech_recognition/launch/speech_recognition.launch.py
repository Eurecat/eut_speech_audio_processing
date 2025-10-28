import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    OpaqueFunction,
    LogInfo,
    SetEnvironmentVariable,
    DeclareLaunchArgument,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Default virtual environment paths for each node type
VENV_PATH_DEFAULT = os.environ.get("AI_VENV", "/opt/ros_python_env")
VENV_PATH_DIARIZATION = os.environ.get("AI_VENV", "/opt/ros_python_diarization_env")


def _venv_site_packages(venv_path: str) -> str:
    """Get site-packages path for a given virtual environment"""
    py = os.path.join(venv_path, "bin", "python")
    return subprocess.check_output(
        [py, "-c", "import site; print(site.getsitepackages()[0])"], text=True
    ).strip()


def _setup(context, *args, **kwargs):
    """Setup function to configure nodes based on launch arguments"""

    # Get launch configuration values
    enable_vad = LaunchConfiguration("enable_vad").perform(context)
    enable_wake_word = LaunchConfiguration("enable_wake_word").perform(context)
    enable_diarization = LaunchConfiguration("enable_diarization").perform(context)
    enable_asr = LaunchConfiguration("enable_asr").perform(context)
    diarization_delay = float(LaunchConfiguration("diarization_delay").perform(context))
    asr_delay = float(LaunchConfiguration("asr_delay").perform(context))

    # Get package config directory
    config_dir = get_package_share_directory("speech_recognition")

    # Common setup for environment variables
    nodes_to_launch = []
    log_messages = []

    # VAD Node Setup
    if enable_vad.lower() == "true":
        site_pkgs_vad = _venv_site_packages(VENV_PATH_DEFAULT)
        existing = os.environ.get("PYTHONPATH", "")
        new_py_path_vad = (
            site_pkgs_vad if not existing else f"{site_pkgs_vad}{os.pathsep}{existing}"
        )
        vad_config_file = os.path.join(config_dir, "config", "vad_params.yaml")

        log_messages.extend(
            [
                LogInfo(
                    msg=f"[speech_recognition] VAD: Using AI venv: {VENV_PATH_DEFAULT}"
                ),
                LogInfo(
                    msg=f"[speech_recognition] VAD: Injecting site-packages: {site_pkgs_vad}"
                ),
                LogInfo(
                    msg=f"[speech_recognition] VAD: Loading config from: {vad_config_file}"
                ),
            ]
        )

        nodes_to_launch.extend(
            [
                SetEnvironmentVariable("PYTHONPATH", new_py_path_vad),
                Node(
                    package="speech_recognition",
                    executable="vad_node",
                    name="vad_node",
                    output="screen",
                    parameters=[vad_config_file],
                    condition=IfCondition(LaunchConfiguration("enable_vad")),
                ),
            ]
        )

    # Wake Word Node Setup
    if enable_wake_word.lower() == "true":
        site_pkgs_wake_word = _venv_site_packages(VENV_PATH_DEFAULT)
        existing = os.environ.get("PYTHONPATH", "")
        new_py_path_wake_word = (
            site_pkgs_wake_word
            if not existing
            else f"{site_pkgs_wake_word}{os.pathsep}{existing}"
        )

        log_messages.extend(
            [
                LogInfo(
                    msg=f"[speech_recognition] Wake Word: Using AI venv: {VENV_PATH_DEFAULT}"
                ),
                LogInfo(
                    msg=f"[speech_recognition] Wake Word: Injecting site-packages: {site_pkgs_wake_word}"
                ),
            ]
        )

        nodes_to_launch.extend(
            [
                SetEnvironmentVariable("PYTHONPATH", new_py_path_wake_word),
                Node(
                    package="speech_recognition",
                    executable="wake_word_node",
                    name="wake_word_node",
                    output="screen",
                    condition=IfCondition(LaunchConfiguration("enable_wake_word")),
                ),
            ]
        )

    # Diarization Node Setup (starts first with 4 second delay)
    if enable_diarization.lower() == "true":
        site_pkgs_diarization = _venv_site_packages(VENV_PATH_DIARIZATION)
        existing = os.environ.get("PYTHONPATH", "")
        new_py_path_diarization = (
            site_pkgs_diarization
            if not existing
            else f"{site_pkgs_diarization}{os.pathsep}{existing}"
        )
        diarization_config_file = os.path.join(
            config_dir, "config", "diarization_params.yaml"
        )

        log_messages.extend(
            [
                LogInfo(
                    msg=f"[speech_recognition] Diarization: Using AI venv: {VENV_PATH_DIARIZATION}"
                ),
                LogInfo(
                    msg=f"[speech_recognition] Diarization: Injecting site-packages: {site_pkgs_diarization}"
                ),
                LogInfo(
                    msg=f"[speech_recognition] Diarization: Loading config from: {diarization_config_file}"
                ),
                LogInfo(
                    msg=f"[speech_recognition] Diarization: Will start with {diarization_delay} second delay"
                ),
            ]
        )

        # Add Diarization node with timer delay (starts first)
        nodes_to_launch.append(
            TimerAction(
                period=diarization_delay,
                actions=[
                    SetEnvironmentVariable("PYTHONPATH", new_py_path_diarization),
                    Node(
                        package="speech_recognition",
                        executable="diarization_node",
                        name="diarization_node",
                        output="screen",
                        parameters=[diarization_config_file],
                        condition=IfCondition(
                            LaunchConfiguration("enable_diarization")
                        ),
                    ),
                ],
            )
        )

    # ASR Node Setup (starts after Diarization with 8 second delay)
    if enable_asr.lower() == "true":
        site_pkgs_asr = _venv_site_packages(VENV_PATH_DEFAULT)
        existing = os.environ.get("PYTHONPATH", "")
        new_py_path_asr = (
            site_pkgs_asr if not existing else f"{site_pkgs_asr}{os.pathsep}{existing}"
        )
        asr_config_file = os.path.join(config_dir, "config", "asr_params.yaml")

        log_messages.extend(
            [
                LogInfo(
                    msg=f"[speech_recognition] ASR: Using AI venv: {VENV_PATH_DEFAULT}"
                ),
                LogInfo(
                    msg=f"[speech_recognition] ASR: Injecting site-packages: {site_pkgs_asr}"
                ),
                LogInfo(
                    msg=f"[speech_recognition] ASR: Loading config from: {asr_config_file}"
                ),
                LogInfo(
                    msg=f"[speech_recognition] ASR: Will start with {asr_delay} second delay"
                ),
            ]
        )

        # Add ASR node with timer delay to start after Diarization
        nodes_to_launch.append(
            TimerAction(
                period=asr_delay,
                actions=[
                    SetEnvironmentVariable("PYTHONPATH", new_py_path_asr),
                    Node(
                        package="speech_recognition",
                        executable="asr_node",
                        name="asr_node",
                        output="screen",
                        parameters=[asr_config_file],
                        condition=IfCondition(LaunchConfiguration("enable_asr")),
                    ),
                ],
            )
        )

    # Combine all log messages and nodes
    return log_messages + nodes_to_launch


def generate_launch_description():
    """Generate the launch description with configurable node enabling/disabling"""

    return LaunchDescription(
        [
            # Declare launch arguments for enabling/disabling nodes
            DeclareLaunchArgument(
                "enable_vad",
                default_value="true",
                description="Enable VAD (Voice Activity Detection) node",
            ),
            DeclareLaunchArgument(
                "enable_wake_word",
                default_value="true",
                description="Enable Wake Word Detection node",
            ),
            DeclareLaunchArgument(
                "enable_diarization",
                default_value="true",
                description="Enable Diarization (Speaker Identification) node",
            ),
            DeclareLaunchArgument(
                "diarization_delay",
                default_value="4.0",
                description="Delay (seconds)",
            ),
            DeclareLaunchArgument(
                "enable_asr",
                default_value="true",
                description="Enable ASR (Automatic Speech Recognition) node",
            ),
            DeclareLaunchArgument(
                "asr_delay",
                default_value="8.0",
                description="Delay (seconds)",
            ),
            # Add informational log message
            LogInfo(msg="[speech_recognition] Starting Speech Recognition Suite"),
            # Setup function to configure and launch nodes
            OpaqueFunction(function=_setup),
        ]
    )
