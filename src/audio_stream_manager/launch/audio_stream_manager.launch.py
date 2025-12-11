import os
import subprocess
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo, SetEnvironmentVariable, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

VENV_PATH = os.environ.get(
    "AI_VENV", "/opt/ros_python_env"
)  # set AI_VENV or uses default


def _venv_site_packages(venv_path: str) -> str:
    py = os.path.join(venv_path, "bin", "python")
    return subprocess.check_output(
        [py, "-c", "import site; print(site.getsitepackages()[0])"], text=True
    ).strip()


def generate_launch_description():
    # Get config file
    config_dir = get_package_share_directory("audio_stream_manager")
    config_file = os.path.join(config_dir, "config", "audio_params.yaml")

    # Load defaults from YAML
    with open(config_file, 'r') as f:
        params_yaml = yaml.safe_load(f)
        
    defaults = params_yaml['audio_capturing_automatic_device_node']['ros__parameters']

    # Setup environment
    site_pkgs = _venv_site_packages(VENV_PATH)
    existing = os.environ.get("PYTHONPATH", "")
    new_py_path = site_pkgs if not existing else f"{site_pkgs}{os.pathsep}{existing}"

    # Declare launch arguments
    launch_args = []
    node_params = {}

    # Define parameters to expose
    params_to_expose = [
        'device_name',
        'dtype',
        'channels',
        'chunk',
        'target_samplerate',
        'disconnection_timeout',
        'disconnection_check_interval',
        'test_stream_duration',
        'primary_device_check_interval'
    ]

    for param in params_to_expose:
        if param in defaults:
            launch_args.append(
                DeclareLaunchArgument(
                    param,
                    default_value=str(defaults[param]),
                    description=f'Parameter {param} from audio_params.yaml'
                )
            )
            node_params[param] = LaunchConfiguration(param)

    return LaunchDescription(
        launch_args +
        [
            LogInfo(msg=f"[audio_stream_manager] Using AI venv: {VENV_PATH}"),
            LogInfo(msg=f"[audio_stream_manager] Injecting site-packages: {site_pkgs}"),
            LogInfo(msg=f"[audio_stream_manager] Loading config from: {config_file}"),
            SetEnvironmentVariable("PYTHONPATH", new_py_path),
            Node(
                package="audio_stream_manager",
                executable="audio_capturing_automatic_device_node",
                name="audio_capturing_automatic_device_node",
                output="screen",
                # Pass config file first, then overrides
                parameters=[config_file, node_params],
            ),
        ]
    )
