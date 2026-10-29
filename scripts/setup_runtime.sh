#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo 'Run sudo bash scripts/install_ros_jazzy.sh first.' >&2
  exit 1
fi
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages .venv-vla
.venv-vla/bin/python -m pip install --upgrade pip 'setuptools>=70,<80' wheel
.venv-vla/bin/python -m pip install -r requirements-vla.txt -c requirements-vla-lock.txt
.venv-vla/bin/python -m pip install cffi
.venv-vla/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is required"; print(torch.cuda.get_device_name())'
HF_HUB_OFFLINE=0 bash scripts/fetch_vla_models.sh
bash scripts/build_water_plugin.sh
(
  cd ros2_ws
  colcon build --packages-select iss_capture_ros --symlink-install
)
echo 'Setup complete: source scripts/ros_env.sh'
echo 'Run: ros2 launch iss_capture_ros one_policy.launch.py task:=all seed:=3300002 record:=true'
