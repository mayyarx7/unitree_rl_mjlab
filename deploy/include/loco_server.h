// The R1 walking service (`sport`) served by this controller, so a client written for Unitree's
// own controller (r1::LocoClient, arena-r1-kit) drives the RL policy unchanged, in simulation
// and on the robot.

#pragma once

#include <unitree/robot/server/server.hpp>
#include <unitree/robot/internal/internal_error.hpp>
#include <unitree/robot/r1/loco/r1_loco_api.hpp>
#include <unitree/robot/r1/loco/r1_loco_error.hpp>
#include <spdlog/spdlog.h>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>

class LocoServer : public unitree::robot::Server
{
public:
    // FSM id as Unitree's controller reports it: 811 is walking, which clients wait for.
    static constexpr int FSM_WALKING = 811;
    // 816: walking with the upper body taken over on rt/arm_sdk; SetVelocity is refused then.
    static constexpr int FSM_ARM_SDK = 816;

    LocoServer() : Server(unitree::robot::r1::LOCO_SERVICE_NAME) {}

    void Init() override
    {
        SetApiVersion(unitree::robot::r1::LOCO_API_VERSION);
        UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(unitree::robot::r1::ROBOT_API_ID_LOCO_GET_FSM_ID, &LocoServer::GetFsmId);
        UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(unitree::robot::r1::ROBOT_API_ID_LOCO_SET_VELOCITY, &LocoServer::SetVelocity);
    }

    // The FSM id to report; set once the FSM exists.
    void set_fsm_id(std::function<int()> fn) { fsm_id_ = std::move(fn); }
    // Whether SetVelocity is taken now (the policy is running).
    void set_accepting(std::function<bool()> fn) { accepting_ = std::move(fn); }

    // The latest velocity while it is valid; zeros and false once its duration has passed.
    bool command(float& vx, float& vy, float& w)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (clock::now() >= until_) {
            vx = vy = w = 0.0f;
            return false;
        }
        vx = vel_[0]; vy = vel_[1]; w = vel_[2];
        return true;
    }

private:
    using clock = std::chrono::steady_clock;

    int32_t GetFsmId(const std::string&, std::string& data)
    {
        if (!fsm_id_) return unitree::robot::r1::UT_ROBOT_LOCO_ERR_LOCOSTATE_NOT_AVAILABLE;
        data = "{\"data\":" + std::to_string(fsm_id_()) + "}";
        return 0;
    }

    int32_t SetVelocity(const std::string& parameter, std::string&)
    {
        unitree::robot::r1::JsonizeVelocityCommand json;
        try {
            unitree::common::FromJsonString(parameter, json);
        } catch (const std::exception&) {
            return unitree::robot::UT_ROBOT_ERR_SERVER_API_PARAMETER;
        }
        if (json.velocity.size() != 3 || !std::isfinite(json.duration) || json.duration < 0.0f) {
            return unitree::robot::UT_ROBOT_ERR_SERVER_API_PARAMETER;
        }
        for (float v : json.velocity) {
            if (!std::isfinite(v)) return unitree::robot::UT_ROBOT_ERR_SERVER_API_PARAMETER;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        // Outside the policy a command is dropped and the last one cleared, so entering it never
        // starts with a stale one.
        if (!accepting_ || !accepting_()) {
            until_ = clock::time_point::min();
            return unitree::robot::r1::UT_ROBOT_LOCO_ERR_LOCOSTATE_NOT_AVAILABLE;
        }
        vel_[0] = json.velocity[0]; vel_[1] = json.velocity[1]; vel_[2] = json.velocity[2];
        until_ = clock::now() + std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(json.duration));
        return 0;
    }

    std::mutex mutex_;
    float vel_[3] = {0.0f, 0.0f, 0.0f};
    clock::time_point until_ = clock::time_point::min();
    std::function<int()> fsm_id_;
    std::function<bool()> accepting_;
};

// The controller's walking service, or null when the joystick commands the velocity.
inline std::shared_ptr<LocoServer> loco_server;
