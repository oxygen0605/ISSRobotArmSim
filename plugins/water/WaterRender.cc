#include "WaterRender.hh"
#include <gz/rendering/DepthCamera.hh>
#include <cmath>
#include <iostream>
#include <functional>
#include <string>
#include <stdexcept>
#include <gz/common/Console.hh>
#include <gz/rendering/Camera.hh>
#include <gz/rendering/Geometry.hh>
#include <gz/rendering/RenderEngine.hh>
#include <gz/rendering/RenderingIface.hh>
#include <gz/rendering/ogre2/Ogre2Geometry.hh>
#include <gz/rendering/ogre2/Ogre2Material.hh>
#include <gz/rendering/ogre2/Ogre2RenderEngine.hh>
#include <gz/rendering/ogre2/Ogre2RenderPass.hh>
#include <OgreMovableObject.h>
#include <OgreItem.h>
#include <OgreSubItem.h>
#include <OgreRoot.h>
#include <OgreHlmsManager.h>
#include <OgreHlms.h>
#include <Hlms/Pbs/OgreHlmsPbsDatablock.h>
#include <Compositor/OgreCompositorManager2.h>
#include <Compositor/OgreCompositorNode.h>
#include <Compositor/OgreCompositorNodeDef.h>
#include <Compositor/OgreCompositorWorkspace.h>
#include <Compositor/Pass/OgreCompositorPass.h>
#include <Compositor/Pass/PassQuad/OgreCompositorPassQuadDef.h>
#include <Compositor/Pass/PassScene/OgreCompositorPassSceneDef.h>

namespace iss {
namespace {
constexpr uint8_t WaterQueue = 210;
constexpr uint32_t WaterDepthVisibility = 1u << 24;

class WaterPass : public gz::rendering::Ogre2RenderPass {
 public:
  WaterPass(const gz::math::Color &background, Ogre::Root *root, std::function<void()> beforeRender,
            std::shared_ptr<unsigned int> created)
      : background(background), root(root), beforeRender(std::move(beforeRender)), created(std::move(created)) {}
  void PreRender(const gz::rendering::CameraPtr &) override { beforeRender(); }
  void CreateRenderPass() override {
    if (!this->ogreCompositorNodeDefName.empty()) return;
    static unsigned int serial = 0;
    this->ogreCompositorNodeDefName = "ISSWaterNode_" + std::to_string(++serial);
    auto manager = root->getCompositorManager2();
    if (const char *debug = std::getenv("ISS_WATER_SHADER_DEBUG"))
      root->getHlmsManager()->getHlms(Ogre::HLMS_PBS)->setDebugOutputPath(true, true, debug);
    // Gazebo invokes CreateRenderPass immediately before recreating its nodes.
    // Edit the base definitions here, so the replacement is installed as part
    // of that rebuild (WorkspaceAdded is not emitted for a late-added pass).
    for (const auto &entry : manager->getNodeDefinitions()) {
      auto base = entry.second;
      if (base->getNameStr().find("/PbsMaterialsRenderingNode") == std::string::npos) continue;
      for (size_t t = 0; t < base->getNumTargetPasses(); ++t) {
        for (auto pass : base->getTargetPass(t)->getCompositorPassesNonConst()) {
          if (pass->getType() != Ogre::PASS_SCENE) continue;
          auto scene = static_cast<Ogre::CompositorPassSceneDef *>(pass);
          scene->mFirstRQ = scene->mLastRQ = 0;
          scene->mShadowNode = Ogre::IdString();
        }
      }
    }
    auto node = manager->addNodeDefinition(this->ogreCompositorNodeDefName);
    node->setNumTargetPass(2);
    node->addTextureSourceName("rt_input", 0, Ogre::TextureDefinitionBase::TEXTURE_INPUT);
    node->addTextureSourceName("rt_output", 1, Ogre::TextureDefinitionBase::TEXTURE_INPUT);
    auto depth = node->addTextureDefinition("water_depth");
    depth->format = Ogre::PFG_D32_FLOAT;
    depth->fsaa = "1";
    auto colour = node->addTextureDefinition("water_background");
    colour->format = Ogre::PFG_UNKNOWN;  // Match the camera's target format.
    colour->fsaa = "1";
    auto backgroundView = node->addRenderTextureView("water_bg_view");
    backgroundView->colourAttachments.resize(1);
    backgroundView->colourAttachments[0].textureName = "water_background";
    backgroundView->depthAttachment.textureName = "water_depth";
    auto outputView = node->addRenderTextureView("water_output_view");
    outputView->colourAttachments.resize(1);
    outputView->colourAttachments[0].textureName = "rt_output";
    outputView->depthAttachment.textureName = "water_depth";

    // Render the actual scene behind the droplet, including its depth. This is
    // the refraction source; it contains no pre-composited blue sphere.
    auto bg = node->addTargetPass("water_bg_view");
    bg->setNumPasses(1);
    auto scene = static_cast<Ogre::CompositorPassSceneDef *>(bg->addPass(Ogre::PASS_SCENE));
    scene->setAllLoadActions(Ogre::LoadAction::Clear);
    scene->setAllClearColours(Ogre::ColourValue(background.R(), background.G(), background.B(), 1));
    scene->setAllStoreActions(Ogre::StoreAction::Store);
    scene->mStoreActionStencil = Ogre::StoreAction::DontCare;
    scene->mFirstRQ = 0;
    scene->mLastRQ = WaterQueue;
    scene->mIncludeOverlays = false;
    scene->setVisibilityMask(~WaterDepthVisibility);
    scene->mShadowNode = "PbsMaterialsShadowNode";

    auto out = node->addTargetPass("water_output_view");
    out->setNumPasses(2);
    auto copy = static_cast<Ogre::CompositorPassQuadDef *>(out->addPass(Ogre::PASS_QUAD));
    copy->mMaterialName = "Ogre/Copy/4xFP32";
    copy->addQuadTextureSource(0, "water_background");
    copy->setAllLoadActions(Ogre::LoadAction::Load);
    copy->setAllStoreActions(Ogre::StoreAction::Store);
    auto refraction = static_cast<Ogre::CompositorPassSceneDef *>(out->addPass(Ogre::PASS_SCENE));
    refraction->mFirstRQ = WaterQueue;
    refraction->mLastRQ = WaterQueue + 1;
    refraction->mIncludeOverlays = false;
    refraction->setVisibilityMask(~WaterDepthVisibility);
    refraction->mShadowNode = "PbsMaterialsShadowNode";
    refraction->setAllLoadActions(Ogre::LoadAction::Load);
    refraction->setAllStoreActions(Ogre::StoreAction::Store);
    refraction->setUseRefractions("water_depth", "water_background");
    node->mapOutputChannel(0, "rt_output");
    node->mapOutputChannel(1, "rt_input");
    ++(*created);
    std::cerr << "[ISS Water] refraction render pass ready" << std::endl;
  }

  void WorkspaceAdded(Ogre::CompositorWorkspace *workspace) override {
    // Gazebo's ordinary scene pass would otherwise draw the refractive surface
    // without a source texture. Replace that scene draw with the pass above.
    // Recreate nodes after modifying definitions, before the first frame.
    bool changed = false;
    for (auto node : workspace->getNodeSequence()) {
      if (node->getDefinition()->getName() == Ogre::IdString(this->ogreCompositorNodeDefName)) continue;
      for (auto pass : node->_getPasses()) {
        auto def = pass->getDefinition();
        if (def->getType() != Ogre::PASS_SCENE) continue;
        auto scene = const_cast<Ogre::CompositorPassSceneDef *>(
            static_cast<const Ogre::CompositorPassSceneDef *>(def));
        if (scene->mFirstRQ >= 254) continue;  // Preserve GUI overlay rendering.
        if (scene->mFirstRQ != 0 || scene->mLastRQ != 0) {
          scene->mFirstRQ = scene->mLastRQ = 0;
          scene->mShadowNode = Ogre::IdString();
          changed = true;
        }
      }
    }
    if (changed) workspace->recreateAllNodes();
    std::cerr << "[ISS Water] refraction compositor connected" << std::endl;
  }
 private:
  gz::math::Color background;
  Ogre::Root *root;
  std::function<void()> beforeRender;
  std::shared_ptr<unsigned int> created;
};
}

void WaterRenderer::Update(double seconds) {
  simulationSeconds = seconds;
  auto engine = gz::rendering::engine("ogre2");
  if (!engine) return;
  for (unsigned int s = 0; s < engine->SceneCount(); ++s) {
    auto scene = engine->SceneByIndex(s);
    if (!scene || !scene->IsInitialized()) continue;
    // All RGB cameras, including the GUI camera, receive the same render pass.
    for (unsigned int c = 0; c < scene->SensorCount(); ++c) {
      if (std::dynamic_pointer_cast<gz::rendering::DepthCamera>(scene->SensorByIndex(c))) continue;
      auto camera = std::dynamic_pointer_cast<gz::rendering::Camera>(scene->SensorByIndex(c));
      if (!camera || cameras.count(camera.get())) continue;
      camera->SetVisibilityMask(~WaterDepthVisibility);
      camera->SetAntiAliasing(0);  // Shared single-sample depth for refraction.
      auto ogreEngine = dynamic_cast<gz::rendering::Ogre2RenderEngine *>(engine);
      if (!ogreEngine || !ogreEngine->OgreRoot()) continue;
      camera->AddRenderPass(std::make_shared<WaterPass>(scene->BackgroundColor(), ogreEngine->OgreRoot(), [this] { ApplySurface(); }, passesCreated));
      cameras.insert(camera.get());
      std::cerr << "[ISS Water] camera attached: " << camera->Name() << std::endl;
    }
    auto visual = water.lock();
    if (!visual) {
      for (unsigned int v = 0; v < scene->VisualCount(); ++v) {
        auto candidate = scene->VisualByIndex(v);
        if (candidate && candidate->Name().find("water_visual") != std::string::npos) {
          visual = candidate;
          water = visual;
          initialScale = visual->LocalScale();
          initialPose = visual->LocalPose();
          std::cerr << "[ISS Water] surface found: " << visual->Name() << std::endl;
          break;
        }
      }
    }
    if (sponge.expired()) {
      for (unsigned int v = 0; v < scene->VisualCount(); ++v) {
        auto candidate = scene->VisualByIndex(v);
        if (candidate && candidate->Name().find("sponge_visual") != std::string::npos && candidate->GeometryCount()) {
          sponge = candidate;
          break;
        }
      }
    }
    if (depthSurface.expired()) {
      for (unsigned int v = 0; v < scene->VisualCount(); ++v) {
        auto candidate = scene->VisualByIndex(v);
        if (candidate && candidate->Name().find("water_depth_sphere") != std::string::npos) {
          depthSurface = candidate;
          break;
        }
      }
    }
    if (!visual) continue;
    ApplySurface();
  }
}

void WaterRenderer::SetWobbleAmplitude(double value) {
    if (!std::isfinite(value) || value < 0 || value > .2)
      throw std::invalid_argument("wobble_amplitude must be between 0 and 0.20");
    wobbleAmplitude = value;
}

void WaterRenderer::ApplySurface() {
    auto visual = water.lock();
    if (!visual) return;
    // A small volume-preserving ellipsoidal surface oscillation. This affects
    // rendering only; it is not SPH, fluid dynamics, or a deformable collision.
    double phase = 2 * 3.141592653589793 * 1.8 * simulationSeconds;
    double sx = 1 + wobbleAmplitude * std::sin(phase);
    double sy = 1 + wobbleAmplitude * std::sin(phase * 1.31 + .6);
    double sz = 1 / (sx * sy);
    auto support = sponge.lock();
    if (capture.active && support) {
      visual->SetWorldPose(AttachedWaterPose(capture, support->WorldPose()));
      // Keep the original sphere at the actual contact offset indefinitely.
      visual->SetLocalScale(initialScale);
      visual->SetVisible(true);
      if (!adhesionReported) {
        std::cerr << "[ISS Water] adhesion_attached start=" << capture.start << std::endl;
        adhesionReported = true;
      }
    } else {
      visual->SetLocalPose(initialPose);
      visual->SetVisible(true);
      visual->SetLocalScale(initialScale * gz::math::Vector3d(sx, sy, sz));
      adhesionReported = false;
    }
    // An explicitly idealized depth return. RGB cameras and their refraction
    // source exclude this visibility layer; the displayed water stays transparent.
    if (auto depth = depthSurface.lock()) {
      depth->SetWorldPose(visual->WorldPose());
      depth->SetLocalScale(initialScale);
      depth->SetVisible(true);
    }
    for (unsigned int g = 0; g < visual->GeometryCount(); ++g) {
      auto geometry = visual->GeometryByIndex(g);
      auto material = std::dynamic_pointer_cast<gz::rendering::Ogre2Material>(geometry->Material());
      auto ogreGeometry = std::dynamic_pointer_cast<gz::rendering::Ogre2Geometry>(geometry);
      if (!material || !ogreGeometry) continue;
      auto block = material->Datablock();
      block->setWorkflow(Ogre::HlmsPbsDatablock::SpecularAsFresnelWorkflow);
      block->setIndexOfRefraction(Ogre::Vector3(1.333f), false);
      block->setRoughness(.045f);
      block->setTransparency(.40f, Ogre::HlmsPbsDatablock::Refractive);
      // Ogre's screen-space strength is an artistic UV-displacement parameter,
      // not an optical IOR. Keep enough displacement to see the small droplet.
      block->setRefractionStrength(.08f);
      ogreGeometry->OgreObject()->setRenderQueueGroup(WaterQueue);
      ogreGeometry->OgreObject()->setCastShadows(false);
    }
}
}
