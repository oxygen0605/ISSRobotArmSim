# Source from an Ubuntu 24.04 shell: source scripts/ros_env.sh
iss_ros_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo 'ROS 2 Jazzy is missing: /opt/ros/jazzy/setup.bash' >&2
  return 1
fi
source /opt/ros/jazzy/setup.bash
if [[ -f "$iss_ros_root/ros2_ws/install/setup.bash" ]]; then
  source "$iss_ros_root/ros2_ws/install/setup.bash"
fi
source "$iss_ros_root/.venv-vla/bin/activate"
source "$iss_ros_root/scripts/water_env.sh"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-93}"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Prefer the NVIDIA-backed D3D12 adapter for Gazebo's WSLg rendering.
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
if [[ -e /dev/dxg ]]; then
  export GALLIUM_DRIVER=d3d12
fi
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
