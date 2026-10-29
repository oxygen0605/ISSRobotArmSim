#pragma once
#include "WaterRender.hh"
#include <gz/gui/Plugin.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/clock.pb.h>
#include <gz/msgs/pose.pb.h>
#include <mutex>
class WaterGui : public gz::gui::Plugin {
  Q_OBJECT
 public:
  void LoadConfig(const tinyxml2::XMLElement *) override;
 protected:
  bool eventFilter(QObject *, QEvent *) override;
 private:
  iss::WaterRenderer renderer;
  std::atomic<double> time{0};
  gz::transport::Node node;
  std::mutex mutex;
  iss::WaterCapture capture;
  bool reported = false;
};
