#!/usr/bin/env bash
# Official Ubuntu Noble / ROS repository only. Does not install or replace models.
set -euo pipefail
if [[ $(id -u) -ne 0 ]]; then
  echo '管理者として実行してください: sudo bash scripts/install_ros_jazzy.sh' >&2
  exit 1
fi
source /etc/os-release
if [[ "$ID" != ubuntu || "$VERSION_ID" != 24.04 ]]; then
  echo 'このスクリプトはUbuntu 24.04専用です。' >&2
  exit 1
fi
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates
iss_source_deb=$(mktemp --suffix=.deb)
trap 'rm -f -- "$iss_source_deb"' EXIT
curl -fL --retry 3 'https://github.com/ros-infrastructure/ros-apt-source/releases/download/1.3.0/ros2-apt-source_1.3.0.noble_all.deb' -o "$iss_source_deb"
echo "f31d84adf5054c7d60ded0e82c0f776a77ea33af53d08f40b9ad7c94cca55296  $iss_source_deb" | sha256sum -c -
dpkg -i "$iss_source_deb"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ros-jazzy-desktop ros-jazzy-ros-gz ros-dev-tools \
  ros-jazzy-gz-ros2-control ros-jazzy-joint-trajectory-controller \
  ros-jazzy-joint-state-broadcaster ros-jazzy-controller-manager ros-jazzy-ros2controlcli \
  python3-venv python3-dev python3-pip build-essential cmake pkg-config \
  libeigen3-dev qtbase5-dev qtdeclarative5-dev libtinyxml2-dev libprotobuf-dev \
  mesa-utils ffmpeg fonts-noto-cjk
