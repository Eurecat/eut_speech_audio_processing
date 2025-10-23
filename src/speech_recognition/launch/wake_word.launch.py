import os
import subprocess

from launch import LaunchDescription
from launch.actions import OpaqueFunction, LogInfo, SetEnvironmentVariable
from launch_ros.actions import Node

VENV_PATH = os.environ.get(
    "AI_VENV", "/opt/ros_python_env"
)  # set AI_VENV or uses default


def _venv_site_packages(venv_path: str) -> str:
    py = os.path.join(venv_path, "bin", "python")
    return subprocess.check_output(
        [py, "-c", "import site; print(site.getsitepackages()[0])"], text=True
    ).strip()


def _setup(context, *args, **kwargs):
    site_pkgs = _venv_site_packages(VENV_PATH)
    existing = os.environ.get("PYTHONPATH", "")
    new_py_path = site_pkgs if not existing else f"{site_pkgs}{os.pathsep}{existing}"

    return [
        LogInfo(msg=f"[speech_recognition] Using AI venv: {VENV_PATH}"),
        LogInfo(msg=f"[speech_recognition] Injecting site-packages: {site_pkgs}"),
        SetEnvironmentVariable("PYTHONPATH", new_py_path),
        Node(
            package="speech_recognition",
            executable="wake_word_node",
            name="wake_word_node",
            output="screen",
        ),
    ]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=_setup)])
