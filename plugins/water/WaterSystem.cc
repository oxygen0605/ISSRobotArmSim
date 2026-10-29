#include "WaterRender.hh"
#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/EventManager.hh>
#include <gz/sim/rendering/Events.hh>
#include <gz/sim/components/ContactSensorData.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/pose.pb.h>
#include <gz/plugin/Register.hh>
#include <iostream>
#include <mutex>

namespace iss {
class WaterSystem : public gz::sim::System, public gz::sim::ISystemConfigure,
                    public gz::sim::ISystemPostUpdate {
 public:
  void Configure(const gz::sim::Entity &, const std::shared_ptr<const sdf::Element> &config,
                 gz::sim::EntityComponentManager &, gz::sim::EventManager &events) override {
    if (config && config->HasElement("wobble_amplitude"))
      renderer.SetWobbleAmplitude(config->Get<double>("wobble_amplitude"));
    if (config) {
      spongeCentre = config->Get<gz::math::Vector3d>("sponge_center", spongeCentre).first;
      robotName = config->Get<std::string>("robot_model", robotName).first;
      gripperName = config->Get<std::string>("sponge_link", gripperName).first;
    }
    publisher = node.Advertise<gz::msgs::Pose>(AdhesionTopic);
    connection = events.Connect<gz::sim::events::PreRender>([this] {
      {
        std::lock_guard<std::mutex> guard(mutex);
        renderer.SetAdhesion(capture);
      }
      renderer.Update(time.load());
      if (renderer.Ready() && !reported) {
        std::cerr << "[ISS Water SERVER] ready wobble_amplitude=" << renderer.WobbleAmplitude()
                  << " adhesion=sphere" << std::endl;
        reported = true;
      }
    });
  }

  void PostUpdate(const gz::sim::UpdateInfo &info, const gz::sim::EntityComponentManager &ecm) override {
    const double seconds = std::chrono::duration<double>(info.simTime).count();
    const double previous = time.exchange(seconds);
    std::lock_guard<std::mutex> guard(mutex);
    if (seconds < previous) capture = WaterCapture{};
    if (!capture.active && !info.paused) {
      bool touching = false;
      ecm.Each<gz::sim::components::ContactSensorData>(
          [&](const gz::sim::Entity &, const gz::sim::components::ContactSensorData *data) {
            const auto name = [&ecm](const gz::msgs::Entity &collision) {
              if (!collision.name().empty()) return collision.name();
              const auto component = ecm.Component<gz::sim::components::Name>(collision.id());
              return component ? component->Data() : std::string();
            };
            for (const auto &contact : data->Data().contact())
              if (IsWaterSpongeContact(name(contact.collision1()), name(contact.collision2()))) touching = true;
            return !touching;
          });
      if (touching) {
        const auto robot = ecm.EntityByComponents(gz::sim::components::Model(), gz::sim::components::Name(robotName));
        const auto drop = ecm.EntityByComponents(gz::sim::components::Model(), gz::sim::components::Name("droplet"));
        const auto gripper = gz::sim::Model(robot).LinkByName(ecm, gripperName);
        if (robot != gz::sim::kNullEntity && drop != gz::sim::kNullEntity && gripper != gz::sim::kNullEntity) {
          const auto support = gz::sim::worldPose(gripper, ecm) * gz::math::Pose3d(spongeCentre, gz::math::Quaterniond::Identity);
          capture = {true, seconds, (support.Inverse()*gz::sim::worldPose(drop, ecm)).Pos()};
          std::cerr << "[ISS Water] adhesion_contact sim_time=" << seconds
                    << " local_centre=" << capture.relativeCentre << std::endl;
        }
      }
    }
    // Rendering state only. This system never writes poses, joints or physics.
    // Repeat the immutable event so a GUI joining late can recover the state.
    const auto wallNow = std::chrono::steady_clock::now();
    if (capture.active && wallNow-lastPublish >= std::chrono::milliseconds(50)) {
      gz::msgs::Pose message;
      message.set_name("sponge_water_adhesion");
      const auto ns = static_cast<int64_t>(std::llround(capture.start*1e9));
      message.mutable_header()->mutable_stamp()->set_sec(ns/1000000000);
      message.mutable_header()->mutable_stamp()->set_nsec(ns%1000000000);
      message.mutable_position()->set_x(capture.relativeCentre.X());
      message.mutable_position()->set_y(capture.relativeCentre.Y());
      message.mutable_position()->set_z(capture.relativeCentre.Z());
      message.mutable_orientation()->set_w(1);
      publisher.Publish(message);
      lastPublish = wallNow;
    }
  }
 private:
  std::string robotName{"so101"}, gripperName{"gripper_link"};
  WaterRenderer renderer;
  std::atomic<double> time{0};
  std::mutex mutex;
  WaterCapture capture;
  gz::math::Vector3d spongeCentre{-.0079, -.000218, -.0981};
  gz::transport::Node node;
  gz::transport::Node::Publisher publisher;
  std::chrono::steady_clock::time_point lastPublish{};
  gz::common::ConnectionPtr connection;
  bool reported = false;
};
}
GZ_ADD_PLUGIN(iss::WaterSystem, gz::sim::System, gz::sim::ISystemConfigure, gz::sim::ISystemPostUpdate)
