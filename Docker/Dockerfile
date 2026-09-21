# Choose the base image at build time. Defaults to Vulcanexus with GPU support.
ARG BASE_IMAGE="eut_ros_torch:humble"
ARG CPU_ONLY="false"
ARG TARGET_DISTRO="humble"

FROM ${BASE_IMAGE}
#https://github.com/Eurecat/EutRobAIDockers

# (Optional) re-declare the args if you want to use them in later instructions
ARG BASE_IMAGE
ARG CPU_ONLY
ARG TARGET_DISTRO

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

# Model weights are never baked into the image. Compose bind-mounts the host
# folder src/speech_recognition/speech_recognition/weights at /workspace/weights;
# every model cache points inside it.
ENV WEIGHTS_DIR=/workspace/weights \
    PYANNOTE_CACHE=/workspace/weights/pyannote \
    HF_HOME=/workspace/weights/huggingface \
    TORCH_HOME=/workspace/weights/torch

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
# --index-strategy unsafe-best-match is required for the CPU path: the CPU
# requirements file adds the https://download.pytorch.org/whl/cpu extra index
# (for torch/torchaudio +cpu wheels), and that index also happens to host an
# old "requests" release. uv's default "first-index" strategy would then pin
# requests to that old version and fail to satisfy diart's requests>=2.31.0.
RUN if [ "$CPU_ONLY" = "true" ]; then \
        /bin/bash -c "source /opt/ros_python_env/bin/activate && \
        uv pip install --index-strategy unsafe-best-match -r requirements_general_active.txt"; \
    else \
        /bin/bash -c "source /opt/ros_python_env/bin/activate && \
        uv pip install -r requirements_general_active.txt && \
        if [ "$TARGET_DISTRO" = "humble" ]; then \
            uv pip install onnxruntime-gpu==1.16.3; \
        else \
            uv pip install onnxruntime-gpu==1.24.1; \
        fi"; \
    fi
# We install onnxruntime-gpu on top (not instead of) the plain "onnxruntime"
# that openwakeword pulls in as a dependency: onnxruntime-gpu's files
# overwrite onnxruntime's in site-packages, so `import onnxruntime` still
# resolves to the GPU build (verified: get_available_providers() lists
# CUDAExecutionProvider), but the "onnxruntime" distribution stays
# registered. Previously this uninstalled "onnxruntime" first, which left
# openwakeword's own declared dependency (onnxruntime<2,>=1.10.0) unsatisfied
# in the environment's package metadata — harmless until something walks
# the full pkg_resources dependency graph and raises DistributionNotFound
# for it (NeMo's model loader does exactly that, see fix_dotted_distinfo.py
# below for the same class of issue with dotted package names).

# NeMo's model loader (nemo.core.classes.common.safe_instantiate, invoked
# from asr_node when asr_backend=parakeet) validates every installed
# package's declared dependencies via the legacy pkg_resources API. That API
# does not apply PEP 503 name normalization to a Requirement string, so a
# dependency declared with a literal dot (diart -> "pyannote.audio>=2.1.1",
# speechbrain's hyperpyyaml -> "ruamel.yaml>=...") fails to match the
# installed distribution when its wheel's dist-info directory was built with
# the now-standard underscored name (pyannote_audio-3.4.0.dist-info) instead
# of the literal dotted one pkg_resources' Requirement.key still expects.
# This only ever surfaces when pkg_resources itself comes from the system
# site-packages (as it does here: the venv has no pkg_resources of its own,
# see asr.launch.py's PYTHONPATH-injection instead of real venv activation)
# AND the process's PATH has the venv's bin/ first (true for every node here,
# since entrypoint.sh activates the venv for the whole container) — nemo
# happens to be the first thing in this stack to exercise that specific
# pkg_resources code path. Fix: duplicate the affected dist-info directories
# under their literal dotted name so pkg_resources finds them either way.
COPY fix_dotted_distinfo.py /tmp/fix_dotted_distinfo.py
RUN /bin/bash -c "source /opt/ros_python_env/bin/activate && \
    python3 /tmp/fix_dotted_distinfo.py \$(python3 -c 'import site; print(site.getsitepackages()[0])')" && \
    rm /tmp/fix_dotted_distinfo.py

# Humble (Python 3.10, /usr/bin/python3 entrypoints) can mix venv matplotlib
# with system mpl_toolkits and crash in diarization.
# Keep system matplotlib pair by removing venv matplotlib only on Humble.
RUN if [ "$TARGET_DISTRO" = "humble" ]; then \
        /bin/bash -c "source /opt/ros_python_env/bin/activate && uv pip uninstall -y matplotlib || true"; \
    fi

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
