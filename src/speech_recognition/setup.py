from setuptools import find_packages, setup

package_name = "speech_recognition"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "torch", "diart"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="joan.omedes@eurecat.org",
    description="TODO: Package description",
    license="TODO: License declaration",
    extras_require={
        "test": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "vad_node = speech_recognition.vad:main",
            "diarization_node = speech_recognition.diarization:main",
        ],
    },
)
