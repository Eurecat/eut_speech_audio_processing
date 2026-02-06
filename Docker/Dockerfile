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

# Copy requirements.txt and install Python dependencies in the virtual environment
# Use CPU or GPU requirements based on build arg
RUN if [ "$CPU_ONLY" = "true" ]; then \
        cp requirements_general_cpu.txt requirements_general_active.txt; \
    else \
        cp requirements_general.txt requirements_general_active.txt; \
    fi


# Install all packages in the single ros_python_env
RUN if [ "$CPU_ONLY" = "true" ]; then \
        /bin/bash -c "source /opt/ros_python_env/bin/activate && \
        uv pip install -r requirements_general_active.txt"; \
    else \
        /bin/bash -c "source /opt/ros_python_env/bin/activate && \
        uv pip install -r requirements_general_active.txt && \
        uv pip uninstall onnxruntime && \
        uv pip install onnxruntime-gpu==1.24.1"; \
    fi
#we need to install onnxruntime separately because openwakeword has a dependency on onnxruntime, and we want to ensure the GPU version is installed if no using cpu flag

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
