# Native water rendering for Gazebo Harmonic

This plugin adds Ogre 2.3 screen-space refraction and a small, volume-preserving
ellipsoidal surface oscillation while the water is free. After an actual
sponge/water collision, the visual remains an unchanged sphere at the initial
contact offset and follows the sponge's pose. It never flattens, shrinks,
disappears, or darkens the sponge. This is a visual attachment, not fluid physics.

The actual scene colour and depth are rendered without the droplet. A separate
refractive pass samples them through the droplet surface. Foreground depth and
screen-edge fallbacks use Ogre's standard refraction shader. This is a real-time
approximation; the displacement strength is artistic and is not a physical IOR.
The 1.333 IOR sets Fresnel reflectance. It is not a two-interface optical ray trace.

Shape animation uses simulation time and scales `(sx, sy, 1/(sx*sy))`, where
`sx=1+A sin(2π·1.8t)` and `sy=1+A sin(2π·1.8·1.31t+0.6)`.
Set `A` with `run_vla.py --wobble-amplitude 0.08`; the default is 0.04 and the
supported range is 0 through 0.20. Zero disables shape oscillation but keeps
refraction. The value is stored in both server and GUI plugin SDF configuration,
checked in their readiness acknowledgements, and recorded in the scene manifest.
No rebuild is needed when changing the value. It is set at startup, not live.
The preview and VLA validation commands support the same option.

Before contact the visual volume is constant. The collision sphere and rigid-body dynamics
remain unchanged; at the default 0.04 the free surface differs from that sphere by
less than 1.1 mm. The conservative gap bound in mm is `12 * (1/(1-A)^2 - 1)`.
The default is 1.6 times the original animation amplitude. The human viewer starts about
22 cm from the droplet, while the policy cameras keep their original positions.
The GUI launcher runs a 1.5-simulation-second observation period before giving
the droplet its upward impulse; `--observe-seconds 0` skips that period.

`WaterSystem` serves sensor cameras and `WaterGui` serves the GUI camera. Both
share `WaterRenderer`. Rendering operations run on their respective render
threads. The Python launcher checks readiness messages, so a missing or broken
plugin cannot silently masquerade as verified refraction.

`WaterSystem` reads the physics contact sensor component and accepts only a
`sponge_collision`/`water_collision` pair. It publishes the first contact's
simulation time and initial droplet centre in the sponge's frame on
`/capture/water_adhesion` (`gz.msgs.Pose`). GUI rendering receives the same
event; ROS bridges it to `geometry_msgs/PoseStamped` for display confirmation.
The visual remains a sphere of its original size and follows the current
sponge pose on every render. The underlying rigid proxy is unchanged: no
physical attachment, pose reset, force or additional controller is added.
Contact success remains physics-based. `run_vla.py` holds the arm after success,
confirms the attached sphere in both renderers and then pauses the simulation.
There is no timed absorption animation or material-color change.

Run `python preview_adhesion.py --follow-demo` to record a fixed preview joint
trajectory using real Gazebo RGB, not VLA predictions. It checks actual contact,
then moves the sponge slightly while the sphere stays attached.
`ctest --test-dir build/water_plugin --output-on-failure` checks the contact
name filter and preservation of the contact offset under translation/rotation.

Build with `bash scripts/build_water_plugin.sh`. The active environment uses
Ubuntu 24.04, ROS 2 Jazzy and Gazebo Harmonic (`scripts/install_ros_jazzy.sh`).
The Ubuntu 22.04 fallback is retained for the historical runtime. The compiler uses system
packages, or the project-local `.runtime` development headers when available.
Gazebo libraries are not patched. Native binaries are generated under `build/`
and are not shipped as portable binaries.

References for the implementation:

- Gazebo rendering plugin examples (Apache-2.0):
  https://github.com/gazebosim/gz-sim/tree/gz-sim8/examples/plugin/rendering_plugins
- Ogre-Next Refractions example and compositor structure (MIT):
  https://github.com/OGRECave/ogre-next/tree/v2-3/Samples/2.0/ApiUsage/Refractions
- Ogre 2.3 `HlmsPbsDatablock::Refractive`, `setUseRefractions`, and shared-depth RTV APIs.

For renderer diagnostics, set `ISS_WATER_SHADER_DEBUG` to an existing directory
ending with `/`. Generated shader sources are diagnostic files, not plugin inputs.
