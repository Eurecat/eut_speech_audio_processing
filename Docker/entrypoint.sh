#!/bin/bash
set -e

echo "=== ENTRYPOINT START $(date) PID=$$ ==="
# Source ROS 2 environment
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    echo "Sourcing ROS 2 environment..."
    source /opt/ros/jazzy/setup.bash
    echo "Sourced ${ROS_DISTRO}"
fi

cd /workspace && colcon build --event-handlers console_direct+

# Source the workspace if it exists
if [ -f "/workspace/install/setup.bash" ]; then
    echo "Sourcing workspace environment..."
    source /workspace/install/setup.bash
fi

# Add ROS sourcing to bashrc for interactive shells
echo "# Auto-source ROS environment" >> /root/.bashrc
echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
echo "source /workspace/install/setup.bash 2>/dev/null || true" >> /root/.bashrc

echo "Python version: $(python3 --version)"
echo "Torch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "=== ENTRYPOINT END $(date) PID=$$ ==="

# Execute the command passed to the container
exec "$@"