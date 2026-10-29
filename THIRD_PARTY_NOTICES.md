# Third-party notices

This file records third-party sources used by this project. It does not license
project-authored material or imply endorsement by the named organizations.

## SO-101 robot description and meshes

Source: TheRobotStudio/SO-ARM100
https://github.com/TheRobotStudio/SO-ARM100
Commit: a758567c3978dfeefe282ede0500085a48fe8f78
License: Apache License 2.0 — see licenses/SO-ARM100-LICENSE.txt and
third_party/SO-ARM100/LICENSE.

The original URDF and 13 referenced STL files are retained unchanged. Generated
simulation descriptions resolve mesh paths, convert URDF to SDF, restore the
training-time material colors, add the sponge or grooved finger pads, and add or
replace simulator controllers/sensors. These modifications are project work.
See the upstream Simulation/SO101/README.md for its model-generation and
Open Duck Mini motor-property attribution. No upstream NOTICE file was found in
the pinned repository tree.

## Gazebo Rendering skybox

The water appearance references the installed Gazebo Rendering 8.2.3 resource
ogre2/media/materials/textures/skybox.dds. Its bytes match the resource from
upstream commit f1249da6e07f0d6b7adaba28109fa7af727fd322.
Repository: https://github.com/gazebosim/gz-rendering
Repository license: Apache License 2.0 — see licenses/gz-rendering-8.2.3-LICENSE.txt.
No separate per-image license/author notice was found. The image is not bundled
in this project's source archive; the normal Gazebo installation supplies it.

## Fonts

The original generated rack/fabric/banner labels use DejaVu Sans and DejaVu Sans
Bold. The font files contain Bitstream Vera and Arev-derived conditions, with
DejaVu changes identified as public domain. See licenses/DejaVuSans.ttf.license.txt,
licenses/DejaVuSans-Bold.ttf.license.txt and licenses/DejaVu-package-copyright.txt.
https://dejavu-fonts.github.io/License.html

The architecture diagram uses Noto Sans CJK JP under SIL Open Font License 1.1.
See licenses/Noto-CJK-package-copyright.txt.
https://github.com/notofonts/noto-cjk/blob/main/Sans/LICENSE

Font binaries are not bundled with the generated PNG/SVG artifacts.

## Model metadata

SmolVLA: lerobot/smolvla_base, revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6.
Processor files: HuggingFaceTB/SmolVLM2-500M-Video-Instruct, revision
7b375e1b73b11138ff12fe22c8f2822d8fe03467.
Both pinned model cards declare Apache-2.0. Copies are under licenses/evidence/;
the common license text is licenses/Apache-2.0.txt. Task-specific LoRA adapters
are trained derivatives; the source models' attribution and conditions remain
relevant to their distribution.

## Reference and names

Kibo cabin reference image:
https://assets.moguravr.com/uploads/2024/12/cbc87ace5d2e20ad6b3e90468aa46c73-kix.ej7onl6a7uq7-jpg.webp

Cabin geometry and textures were generated in this project. The reference image
and the third-party ISS_KIBOU/Fab asset are not included as runtime assets.
The reference image's republication license has not been verified.
The generated banner now reads JAKUSA, as requested on 2026-10-26. Earlier
previews may show JAXA. No official-logo bitmap is used. Font permission does
not by itself establish permission for names/trademarks.
This is an independent educational simulation, not an official JAXA product.
