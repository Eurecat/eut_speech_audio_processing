# Choose the base image at build time. Defaults to Vulcanexus.
ARG BASE_IMAGE="eut_ros_vulcanexus_torch:jazzy"

FROM ${BASE_IMAGE}
#https://github.com/Eurecat/EutRobAIDockers

# (Optional) re-declare the arg if you want to use it in later instructions
ARG BASE_IMAGE

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
 
# Install this for audio dependencies
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libasound2-dev \
    libpulse-dev \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
    ffmpeg \
    python3-pytest python3-pytest-cov \
&& rm -rf /var/lib/apt/lists/*
 
# Set working directory
WORKDIR /workspace

# Copy requirements.txt and install Python dependencies in the virtual environment
COPY requirements_general.txt /workspace/requirements_general.txt
RUN /bin/bash -c "source /opt/ros_python_env/bin/activate && uv pip install --no-cache -r requirements_general.txt"

# Create a new virtual environment for diarization and install requirements
RUN python3 -m venv /opt/ros_python_diarization_env
COPY requirements_diarization.txt /workspace/requirements_diarization.txt
RUN /bin/bash -c "source /opt/ros_python_diarization_env/bin/activate && pip install --no-cache-dir -r requirements_diarization.txt"

# ------------------------------------------------------------
# Setup rosdep
# (init system-wide, update for this user)
# ------------------------------------------------------------

# Optional: fix rosdep init issue gracefully
RUN [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ] /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash;sudo rosdep init" || echo "rosdep already initialized"
RUN apt-get update && \ 
    /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash;\
      rosdep update;"
 
# Copy necessary files to the workspace
COPY deps/hri_msgs /workspace/src/hri_msgs
# Copy only audio_common_msgs from audio_common repository
COPY deps/audio_common/audio_common_msgs /workspace/src/audio_common_msgs
# COPY deps/eut_yolo/src/eut_yolo_core /workspace/src/eut_yolo_core
# COPY deps/eut_yolo/src/eut_yolo_interfaces /workspace/src/eut_yolo_interfaces

# ------------------------------------------------------------
# Install dependencies
# (from workspace package.xmls via rosdep)
# ------------------------------------------------------------
RUN rosdep install --from-paths src --ignore-src -r -y
 
# ------------------------------------------------------------
# Build workspace
# ------------------------------------------------------------
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash; colcon build --event-handlers console_direct+ --symlink-install;"

# Copy coverage script for automated testing within CI/CD
COPY ci_cd_coverage.sh /ci_cd_coverage.sh
RUN chmod +x /ci_cd_coverage.sh


# Copy and set up the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]
