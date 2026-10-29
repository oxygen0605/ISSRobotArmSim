"""Native Gazebo/Ogre2 materials; the collision sphere is never modified."""
import os
from pathlib import Path
import xml.etree.ElementTree as ET

APPEARANCES = ("refractive", "water", "training")


def element(parent, tag, text):
    node = ET.SubElement(parent, tag)
    node.text = str(text)
    return node


def reflection_map():
    """Use the reflection cubemap already distributed with Gazebo Rendering."""
    roots = []
    if os.environ.get("GZ_RENDERING_RESOURCE_PATH"):
        roots.extend(Path(p) for p in os.environ['GZ_RENDERING_RESOURCE_PATH'].split(os.pathsep) if p)
    roots.extend(Path('/opt/ros/jazzy/opt').glob('gz_rendering_vendor/share/gz/gz-rendering8'))
    roots.extend([Path("/usr/share/gz/gz-rendering8"), Path("/usr/local/share/gz/gz-rendering8")])
    for root in roots:
        path = root / "ogre2/media/materials/textures/skybox.dds"
        if path.is_file():
            return path.resolve()
    return None


def apply_droplet_appearance(world, appearance):
    if appearance not in APPEARANCES:
        raise ValueError(f"水滴の外観は{APPEARANCES}から選択してください。")
    if appearance == "training":
        return {"profile": "training", "version": 1, "transparent": False}
    visual = world.find("model[@name='droplet']/link[@name='body']/visual[@name='water_visual']")
    material = visual.find("material")
    material.clear()
    # A blue tint preserves recognition by the existing policy. Transparency is
    # applied once at visual level, not multiplied again through diffuse alpha.
    element(material, "ambient", "0.08 0.48 0.68 1")
    element(material, "diffuse", "0.12 0.62 0.90 1")
    element(material, "specular", "0.90 0.96 1.0 1")
    element(material, "shininess", 160)
    metal = ET.SubElement(ET.SubElement(material, "pbr"), "metal")
    element(metal, "metalness", 0.0)
    element(metal, "roughness", 0.055)
    environment = reflection_map()
    if environment:
        element(metal, "environment_map", environment.as_uri())
    element(visual, "transparency", 0.55)
    element(visual, "cast_shadows", "false")
    sun = world.find("light[@name='sun']/specular")
    if sun is not None:
        sun.text = "0.9 0.95 1 1"
    return {"profile": appearance, "version": 1, "transparent": True,
            "transparency": 0.55, "roughness": 0.055, "metalness": 0.0,
            "reflection_map": "gazebo_bundled_skybox" if environment else None,
            "optics": "native screen-space refraction plugin" if appearance == 'refractive' else "PBR reflection + alpha transparency; no refractive ray tracing"}
