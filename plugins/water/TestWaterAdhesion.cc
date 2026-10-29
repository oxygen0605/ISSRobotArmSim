#include "WaterAdhesion.hh"
#include <stdexcept>
#include <iostream>

void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
int main() {
  using namespace iss;
  check(IsWaterSpongeContact("so101::gripper_link::sponge_collision", "droplet::body::water_collision"), "real pair");
  check(IsWaterSpongeContact("water_collision", "sponge_collision"), "reversed pair");
  check(!IsWaterSpongeContact("pen_collision", "sponge_collision"), "pen must not trigger water adhesion");
  check(!IsWaterSpongeContact("water_collision_extra", "sponge_collision"), "partial name must not trigger");
  WaterCapture capture{true, 5., {.03,-.04,.01}};
  auto pose=AttachedWaterPose(capture, gz::math::Pose3d::Zero);
  check(pose.Pos().Equal(capture.relativeCentre, 1e-10), "contact offset must not be pulled inward");
  const gz::math::Pose3d movingSponge(1,2,3,0,0,1.5707963267948966);
  pose=AttachedWaterPose(capture, movingSponge);
  check(pose.Pos().Equal({1.04,2.03,3.01}, 1e-10), "sphere must follow translation and rotation");
  check((movingSponge.Inverse()*pose).Pos().Equal(capture.relativeCentre,1e-10), "relative contact position must remain fixed");
  std::cout << "Water adhesion contact and rigid-follow checks passed\n";
}
