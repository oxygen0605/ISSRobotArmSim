"""Attach the native refraction/wobble plugins, failing clearly if unbuilt."""
import argparse
import hashlib
import math
import os
import xml.etree.ElementTree as ET
from .config import ROOT, SPONGE_CENTER

DEFAULT_WOBBLE_AMPLITUDE = .04


def validate_wobble_amplitude(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= .2:
        raise ValueError('揺らぎの振幅は0〜0.20で指定してください（標準0.04）。')
    return value if value else 0.0


def add_wobble_argument(parser):
    def checked(value):
        try:
            return validate_wobble_amplitude(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    parser.add_argument('--wobble-amplitude', type=checked, default=DEFAULT_WOBBLE_AMPLITUDE,
                        help='表面の揺らぎの振幅: 0〜0.20、標準0.04。refractive表示に適用')


def install_water_plugins(root, wobble_amplitude=DEFAULT_WOBBLE_AMPLITUDE):
    amplitude = validate_wobble_amplitude(wobble_amplitude)
    directory = ROOT / 'build/water_plugin'
    system = directory / 'libiss-water-system.so'
    gui_library = directory / 'libWaterGui.so'
    if not system.is_file() or not gui_library.is_file():
        raise RuntimeError('屈折プラグインがありません。bash scripts/build_water_plugin.sh を実行してください。')
    world = root.find('world')
    server = ET.SubElement(world, 'plugin', filename=str(system), name='iss::WaterSystem')
    ET.SubElement(server, 'wobble_amplitude').text = str(amplitude)
    gui = world.find('gui')
    # gz-gui derives its Qt resource URL from the library name. An absolute
    # filename would turn that URL into an invalid resource path.
    plugin = ET.SubElement(gui, 'plugin', filename='WaterGui', name='WaterGui')
    ET.SubElement(plugin, 'wobble_amplitude').text = str(amplitude)
    ET.SubElement(server, 'sponge_center').text = ' '.join(map(str, SPONGE_CENTER))
    settings = ET.SubElement(plugin, 'gz-gui')
    ET.SubElement(settings, 'property', key='showTitleBar', type='bool').text = 'false'
    ET.SubElement(settings, 'property', key='visible', type='bool').text = 'false'
    return {'mode': 'ogre2_screen_space_refraction', 'version': 6, 'ior_for_fresnel': 1.333,
            'refraction_strength': .08, 'wobble_amplitude': amplitude, 'wobble_frequency_hz': 1.8,
            'volume_preserving_visual_scale': True,
            'collision_shape': 'unchanged sphere',
            'adhesion': {'enabled': True, 'shape': 'sphere', 'preserve_contact_offset': True,
                           'trigger': 'actual sponge_collision / water_collision contact',
                           'state_topic': '/capture/water_adhesion', 'visual_only': True,
                           'physics_attachment': False, 'fluid_solver': False},
            'system_plugin_sha256': hashlib.sha256(system.read_bytes()).hexdigest(),
            'gui_plugin_sha256': hashlib.sha256(gui_library.read_bytes()).hexdigest()}


def configure_water_plugin_paths():
    directory = str(ROOT / 'build/water_plugin')
    paths = os.environ.get('GZ_GUI_PLUGIN_PATH', '').split(os.pathsep)
    if directory not in paths:
        os.environ['GZ_GUI_PLUGIN_PATH'] = os.pathsep.join([directory, *filter(None, paths)])
