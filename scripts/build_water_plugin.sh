#!/usr/bin/env bash
# Build a Harmonic / Ogre 2.3 plugin against either system packages or the
# project-local validation runtime. No Gazebo source or binary is patched.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  set +u
  source /opt/ros/jazzy/setup.bash
  set -u
  cmake -S plugins/water -B build/water_plugin -DCMAKE_BUILD_TYPE=Release
  cmake --build build/water_plugin --parallel 4
  exit 0
fi
if [[ -f .runtime/root/usr/include/gz/sim8/gz/sim/System.hh ]]; then
  source .runtime/enable.sh
  iss_prefix="$PWD/.runtime/root/usr"
else
  iss_prefix=/usr
fi
iss_output="$PWD/build/water_plugin"
mkdir -p "$iss_output"
iss_stage=$(mktemp -d "$iss_output/.build.XXXXXX")
trap 'rm -rf -- "$iss_stage"' EXIT
iss_flags=(-std=c++17 -O2 -DNDEBUG -fPIC -pthread -Wall -Wextra -Wno-unused-parameter -Iplugins/water -I"$iss_prefix/include" -I"$iss_prefix/include/eigen3")
for d in "$iss_prefix"/include/gz/*; do iss_flags+=(-I"$d"); done
iss_flags+=(-I"$iss_prefix/include/OGRE-2.3" -I"$iss_prefix/include/OGRE-2.3/Hlms/Pbs" -I"$iss_prefix/include/OGRE-2.3/Hlms/Unlit" -I"$iss_prefix/include/OGRE-2.3/Hlms/Common")
iss_libs=(-L"$iss_prefix/lib/x86_64-linux-gnu" -L"$iss_prefix/lib/x86_64-linux-gnu/OGRE-2.3" -lgz-rendering8-ogre2 -lgz-rendering8 -lgz-common5 -lgz-common5-events -lgz-math7 -lgz-plugin2 -lOgreNextMain -lOgreNextHlmsPbs)
g++ "${iss_flags[@]}" -c plugins/water/WaterRender.cc -o "$iss_stage/WaterRender.o"
g++ "${iss_flags[@]}" -shared plugins/water/WaterSystem.cc "$iss_stage/WaterRender.o" "${iss_libs[@]}" -lgz-sim8 -lgz-transport13 -lgz-msgs10 -lprotobuf -lsdformat14 -o "$iss_stage/libiss-water-system.so"
for d in "$iss_prefix/include/x86_64-linux-gnu/qt5" "$iss_prefix"/include/x86_64-linux-gnu/qt5/Qt*; do iss_flags+=(-I"$d"); done
iss_moc_flags=()
for flag in "${iss_flags[@]}"; do if [[ "$flag" == -I* || "$flag" == -D* ]]; then iss_moc_flags+=("$flag"); fi; done
"$iss_prefix/lib/qt5/bin/moc" "${iss_moc_flags[@]}" plugins/water/WaterGui.hh -o "$iss_stage/moc_WaterGui.cc"
"$iss_prefix/lib/qt5/bin/rcc" -name WaterGui plugins/water/WaterGui.qrc -o "$iss_stage/qrc_WaterGui.cc"
g++ "${iss_flags[@]}" -shared plugins/water/WaterGui.cc "$iss_stage/WaterRender.o" "$iss_stage/moc_WaterGui.cc" "$iss_stage/qrc_WaterGui.cc" "${iss_libs[@]}" -lgz-gui8 -lgz-transport13 -lgz-msgs10 -lprotobuf -lQt5Core -lQt5Gui -lQt5Quick -lQt5Qml -ltinyxml2 -o "$iss_stage/libWaterGui.so"
# Replace completed libraries atomically; do not truncate a library in use.
mv -f "$iss_stage/libiss-water-system.so" "$iss_output/libiss-water-system.so"
mv -f "$iss_stage/libWaterGui.so" "$iss_output/libWaterGui.so"
echo "Built: $iss_output"
