# Choose the base image at build time. Defaults to Vulcanexus with GPU support.
ARG BASE_IMAGE="eut_ros_vulcanexus_torch:jazzy"
ARG CPU_ONLY="false"

FROM ${BASE_IMAGE}
#https://github.com/Eurecat/EutRobAIDockers

# (Optional) re-declare the args if you want to use them in later instructions
ARG BASE_IMAGE
ARG CPU_ONLY

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
 
# Install CUDA dependencies only if not CPU_ONLY
RUN if [ "$CPU_ONLY" = "false" ]; then \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
        dpkg -i cuda-keyring_1.1-1_all.deb; \
    fi

# Install audio dependencies (conditionally install CUDA libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libasound2-dev \
    libpulse-dev \
    $(if [ "$CPU_ONLY" = "false" ]; then echo "libcudnn9-cuda-12 libcudnn9-dev-cuda-12"; fi) \
    ffmpeg \
    python3-pytest python3-pytest-cov \
&& rm -rf /var/lib/apt/lists/*
 
# Set working directory
WORKDIR /workspace

# Copy all requirements files
COPY requirements_general.txt /workspace/requirements_general.txt
COPY requirements_general_cpu.txt /workspace/requirements_general_cpu.txt
COPY requirements_diarization.txt /workspace/requirements_diarization.txt
COPY requirements_diarization_cpu.txt /workspace/requirements_diarization_cpu.txt

# Copy requirements.txt and install Python dependencies in the virtual environment
# Use CPU or GPU requirements based on build arg
RUN if [ "$CPU_ONLY" = "true" ]; then \
        cp requirements_general_cpu.txt requirements_general_active.txt; \
    else \
        cp requirements_general.txt requirements_general_active.txt; \
    fi

# Install PyTorch with correct variant (CPU or GPU) and clean up
RUN if [ "$CPU_ONLY" = "true" ]; then \
        /bin/bash -c "source /opt/ros_python_env/bin/activate && pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.10.0+cpu torchaudio==2.10.0+cpu --no-cache-dir && pip cache purge"; \
    fi

RUN /bin/bash -c "source /opt/ros_python_env/bin/activate && uv pip install --no-cache -r requirements_general_active.txt" && \
    rm -rf /root/.cache/pip /tmp/* /var/tmp/*

# Create a new virtual environment for diarization and install requirements
RUN python3 -m venv /opt/ros_python_diarization_env

# Copy appropriate diarization requirements
RUN if [ "$CPU_ONLY" = "true" ]; then \
        cp requirements_diarization_cpu.txt requirements_diarization_active.txt; \
    else \
        cp requirements_diarization.txt requirements_diarization_active.txt; \
    fi

# Install PyTorch for diarization environment with space optimization
RUN if [ "$CPU_ONLY" = "true" ]; then \
        /bin/bash -c "source /opt/ros_python_diarization_env/bin/activate && pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchaudio --no-cache-dir && pip cache purge"; \
    fi

RUN /bin/bash -c "source /opt/ros_python_diarization_env/bin/activate && pip install --no-cache-dir -r requirements_diarization_active.txt" && \
    rm -rf /root/.cache/pip /tmp/* /var/tmp/* && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

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

# Copy coverage script for automated testing within CI/CD
COPY quick_test_coverage.sh /quick_test_coverage.sh
RUN chmod +x /quick_test_coverage.sh

# Copy and set up the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]
