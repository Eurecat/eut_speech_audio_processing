# Choose the base image at build time. Defaults to Vulcanexus.
ARG BASE_IMAGE="eut_ros_vulcanexus_torch:jazzy"

FROM ${BASE_IMAGE}
#https://github.com/Eurecat/EutRobAIDockers

# (Optional) re-declare the arg if you want to use it in later instructions
ARG BASE_IMAGE

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
 
# ------------------------------------------------------------
# Install system dependencies & development tools
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-rosdep \
    python3-colcon-common-extensions \
    python3-vcstool \
    build-essential \
    ros-${ROS_DISTRO}-usb-cam \
    git \
    portaudio19-dev \
    libasound2-dev \
    libpulse-dev \
&& rm -rf /var/lib/apt/lists/*
 
# Set working directory
WORKDIR /workspace

# Copy requirements.txt and install Python dependencies using uv
COPY requirements.txt /workspace/requirements.txt
RUN uv pip install --system --no-cache --break-system-packages -r requirements.txt

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
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash; colcon build --event-handlers console_direct+;"


# Copy and set up the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]
