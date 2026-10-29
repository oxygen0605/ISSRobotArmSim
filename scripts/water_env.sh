# Source this before manually launching a generated world with `gz sim`.
# Python launchers configure the same path automatically.
iss_water_project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GZ_GUI_PLUGIN_PATH="$iss_water_project/build/water_plugin:${GZ_GUI_PLUGIN_PATH:-}"
