#include "WaterGui.hh"
#include <gz/gui/Application.hh>
#include <gz/gui/MainWindow.hh>
#include <gz/gui/GuiEvents.hh>
#include <gz/plugin/Register.hh>
#include <iostream>
#include <tinyxml2.h>
#include <stdexcept>
void WaterGui::LoadConfig(const tinyxml2::XMLElement *config) {
  if (config) {
    if (auto element = config->FirstChildElement("wobble_amplitude")) {
      double amplitude = .04;
      if (element->QueryDoubleText(&amplitude) != tinyxml2::XML_SUCCESS)
        throw std::invalid_argument("Invalid wobble_amplitude");
      renderer.SetWobbleAmplitude(amplitude);
    }
  }
  gz::gui::App()->findChild<gz::gui::MainWindow *>()->installEventFilter(this);
  node.Subscribe<gz::msgs::Clock>("/world/microgravity/clock", [this](const gz::msgs::Clock &clock) {
    const double now = clock.sim().sec() + clock.sim().nsec() * 1e-9;
    if (now < time.exchange(now)) {
      std::lock_guard<std::mutex> guard(mutex);
      capture = iss::WaterCapture{};
    }
  });
  node.Subscribe<gz::msgs::Pose>(iss::AdhesionTopic, [this](const gz::msgs::Pose &message) {
    if (message.name() != "sponge_water_adhesion") return;
    const double start = message.header().stamp().sec()+message.header().stamp().nsec()*1e-9;
    if (start > time.load()+.1) return;  // Old event after a world reset.
    std::lock_guard<std::mutex> guard(mutex);
    capture = {true, start,
               {message.position().x(), message.position().y(), message.position().z()}};
  });
}
bool WaterGui::eventFilter(QObject *object, QEvent *event) {
  if (event->type() == gz::gui::events::Render::kType) {
    {
      std::lock_guard<std::mutex> guard(mutex);
      renderer.SetAdhesion(capture);
    }
    renderer.Update(time.load());
    if (renderer.Ready() && !reported) {
      std::cerr << "[ISS Water GUI] ready wobble_amplitude=" << renderer.WobbleAmplitude()
                << " adhesion=sphere" << std::endl;
      reported = true;
    }
  }
  return QObject::eventFilter(object, event);
}
GZ_ADD_PLUGIN(WaterGui, gz::gui::Plugin)
