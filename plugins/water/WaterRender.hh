#pragma once
#include <atomic>
#include <memory>
#include <unordered_set>
#include <gz/math/Vector3.hh>
#include <gz/math/Pose3.hh>
#include <gz/rendering/Scene.hh>
#include <gz/rendering/Visual.hh>
#include "WaterAdhesion.hh"

namespace iss {
// Called only on the rendering thread. Simulation time is passed atomically by
// the server/GUI wrappers, so pause and slow simulation behave consistently.
class WaterRenderer {
 public:
  void Update(double simSeconds);
  void SetWobbleAmplitude(double value);
  void SetAdhesion(const WaterCapture &value) { capture = value; }
  double WobbleAmplitude() const { return wobbleAmplitude; }
  bool Ready() const { return !water.expired() && !sponge.expired() && !cameras.empty() && *passesCreated >= cameras.size(); }
 private:
  void ApplySurface();
  double simulationSeconds = 0;
  double wobbleAmplitude = .04;
  std::weak_ptr<gz::rendering::Visual> water;
  std::weak_ptr<gz::rendering::Visual> depthSurface;
  std::weak_ptr<gz::rendering::Visual> sponge;
  gz::math::Pose3d initialPose;
  WaterCapture capture;
  bool adhesionReported = false;
  gz::math::Vector3d initialScale{1, 1, 1};
  std::unordered_set<const void *> cameras;
  std::shared_ptr<unsigned int> passesCreated = std::make_shared<unsigned int>(0);
};
}
