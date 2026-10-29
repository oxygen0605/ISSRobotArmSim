#pragma once
#include <string>
#include <gz/math/Vector3.hh>
#include <gz/math/Pose3.hh>

namespace iss {
constexpr const char *AdhesionTopic = "/capture/water_adhesion";
struct WaterCapture {
  bool active = false;
  double start = 0;
  // Initial droplet centre in the sponge visual's local coordinate frame.
  gz::math::Vector3d relativeCentre{0, 0, 0};
};
inline bool IsWaterSpongeContact(const std::string &a, const std::string &b) {
  const auto ends = [](const std::string &value, const std::string &name) {
    return value == name || (value.size() > name.size()+2 &&
        value.compare(value.size()-name.size()-2, name.size()+2, "::"+name) == 0);
  };
  return (ends(a, "sponge_collision") && ends(b, "water_collision")) ||
         (ends(b, "sponge_collision") && ends(a, "water_collision"));
}
inline gz::math::Pose3d AttachedWaterPose(const WaterCapture &capture,
    const gz::math::Pose3d &spongePose) {
  // Keep the first actual contact offset. No inward pull or time-based shrink.
  return spongePose * gz::math::Pose3d(capture.relativeCentre, gz::math::Quaterniond::Identity);
}
}
