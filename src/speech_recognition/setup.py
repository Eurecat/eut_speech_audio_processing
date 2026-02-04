from setuptools import find_packages, setup

package_name = "speech_recognition"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["launch/vad.launch.py"]),
        ("share/" + package_name, ["launch/diarization.launch.py"]),
        ("share/" + package_name, ["launch/asr.launch.py"]),
        ("share/" + package_name, ["launch/speech_recognition.launch.py"]),
        ("share/" + package_name, ["launch/wake_word.launch.py"]),
        ("share/" + package_name + "/config", ["config/asr_params.yaml"]),
        ("share/" + package_name + "/config", ["config/diarization_params.yaml"]),
        ("share/" + package_name + "/config", ["config/vad_params.yaml"]),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "torch",
        "diart",
        "openwakeword",
        "faster-whisper",
    ],
    zip_safe=True,
    maintainer="root",
    maintainer_email="joan.omedes@eurecat.org",
    description="Speech recognition and processing for ROS2",
    license="Apache-2.0",
    tests_require=['pytest'],
    entry_points={
        "console_scripts": [
            "vad = speech_recognition.vad:main",
            "vad_node = speech_recognition.vad:main",
            "diarization = speech_recognition.diarization:main",
            "diarization_node = speech_recognition.diarization:main",
            "asr = speech_recognition.asr:main", 
            "asr_node = speech_recognition.asr:main",
            "wake_word = speech_recognition.wake_word:main",
            "wake_word_node = speech_recognition.wake_word:main",
        ],
    },
)
