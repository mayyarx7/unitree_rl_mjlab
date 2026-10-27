// rt/heightmap for the perceptive policy's height_scan observation. The map comes from the
// perception process (unitree_rl_mjlab/perception/heightmap_node.py), in simulation and on the
// robot: unitree_go HeightMap_, rows along y with x within a row, each value the pelvis height
// minus the ground height in metres, as Unitree-R1-Perceptive's training scan.
//
// The observation checks the map's layout against its deploy.yaml params and its age (by arrival:
// the perception process may run on another computer's clock). Without a fitting map younger than
// max_age_s it returns flat ground (fallback_height everywhere) and reports it; the controller
// then takes no walking commands, so the robot stands until maps arrive again.

#pragma once

#include <unitree/idl/go2/HeightMap_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <spdlog/spdlog.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>

#include "isaaclab/manager/observation_manager.h"

class HeightmapInput
{
public:
    explicit HeightmapInput(const std::string& topic = "rt/heightmap")
    : sub_(topic)
    {
        sub_.InitChannel([this](const void* msg) {
            std::lock_guard<std::mutex> lock(mutex_);
            map_ = *static_cast<const unitree_go::msg::dds_::HeightMap_*>(msg);
            at_ = clock::now();
            have_ = true;
        }, 1);
        spdlog::info("height_scan: listening on {}", topic);
    }

    struct Layout
    {
        uint32_t width, height;
        float resolution, origin_x, origin_y;
    };

    // The latest map's data if it has this layout and arrived at most max_age_s ago.
    bool get(const Layout& layout, double max_age_s, std::vector<float>& out)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!have_) return note(false, "no height map yet");
        if (map_.width() != layout.width || map_.height() != layout.height ||
            std::abs(map_.resolution() - layout.resolution) > 1e-4f ||
            std::abs(map_.origin()[0] - layout.origin_x) > 1e-4f ||
            std::abs(map_.origin()[1] - layout.origin_y) > 1e-4f ||
            map_.data().size() != size_t(layout.width) * layout.height) {
            return note(false, "height map layout differs from the policy's");
        }
        const double age = std::chrono::duration<double>(clock::now() - at_).count();
        if (age > max_age_s) return note(false, fmt::format("height map older than {:g} s", max_age_s));
        out.assign(map_.data().begin(), map_.data().end());
        return note(true, "");
    }

    // Whether the last observation had to fall back to flat ground.
    bool falling_back() const { return falling_back_; }

private:
    using clock = std::chrono::steady_clock;

    // Logs each change between using maps and falling back, not every step.
    bool note(bool ok, const std::string& why)
    {
        if (!ok && !falling_back_) spdlog::warn("height_scan: {}; flat ground, walking commands refused", why);
        if (ok && falling_back_) spdlog::info("height_scan: height maps arriving, walking commands taken again");
        falling_back_ = !ok;
        return ok;
    }

    unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::HeightMap_> sub_;
    std::mutex mutex_;
    unitree_go::msg::dds_::HeightMap_ map_;
    clock::time_point at_;
    bool have_ = false;
    std::atomic<bool> falling_back_{false};
};

// Made on the first height_scan observation, so a policy without one never subscribes.
inline std::shared_ptr<HeightmapInput> heightmap_input;

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(height_scan)
{
    const HeightmapInput::Layout layout{
        params["width"].as<uint32_t>(), params["height"].as<uint32_t>(), params["resolution"].as<float>(),
        params["origin"][0].as<float>(), params["origin"][1].as<float>()};
    if (!heightmap_input) heightmap_input = std::make_shared<HeightmapInput>(params["topic"].as<std::string>("rt/heightmap"));
    std::vector<float> scan;
    if (!heightmap_input->get(layout, params["max_age_s"].as<double>(0.3), scan)) {
        scan.assign(size_t(layout.width) * layout.height, params["fallback_height"].as<float>(0.74f));
    }
    return scan;
}

} // namespace mdp
} // namespace isaaclab
