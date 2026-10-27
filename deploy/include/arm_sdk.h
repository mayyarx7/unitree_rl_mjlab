// rt/arm_sdk as Unitree's own controller takes it, so a client written for it (arena-r1-kit's
// head and arm teleop) drives this controller unchanged, in simulation and on the robot.
//
// A frame carries targets (q, kp, kd) for the upper body and a weight in mode_pr (0-100). Each
// cycle the weight blends the frame's command over this controller's own for those joints: the
// policy's action where the policy drives the joint, the held command (FixStand's) elsewhere
// (the head). Frames whose mode_machine is not the robot's are dropped. Frames older than
// STALE_S ramp the weight out over RAMP_OUT_S, so a lost client hands the joints back.

#pragma once

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>

class ArmSdk
{
public:
    // Both arms, waist yaw and head: the joints arm_sdk takes on the R1.
    static constexpr std::array<int, 13> JOINTS = {15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 13, 29, 30};
    static constexpr double STALE_S = 0.5;
    static constexpr double RAMP_OUT_S = 1.5;

    explicit ArmSdk(const std::string& topic = "rt/arm_sdk")
    : sub_(topic)
    {
        sub_.InitChannel([this](const void* msg) {
            std::lock_guard<std::mutex> lock(mutex_);
            frame_ = *static_cast<const unitree_hg::msg::dds_::LowCmd_*>(msg);
            at_ = clock::now();
            fresh_ = true;
        }, 1);
    }

    // Blends the latest frame into cmd; driven lists the slots the policy writes this cycle.
    void apply(unitree_hg::msg::dds_::LowCmd_& cmd, const std::vector<float>& driven, uint8_t mode_machine)
    {
        unitree_hg::msg::dds_::LowCmd_ frame;
        double age;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (fresh_ && frame_.mode_machine() != mode_machine) {
                if (!warned_) spdlog::warn("arm_sdk: dropping frames with mode_machine {} (robot {})",
                                           frame_.mode_machine(), mode_machine);
                warned_ = true;
                fresh_ = false;
            }
            if (fresh_) { last_ = frame_; last_at_ = at_; fresh_ = false; have_ = true; }
            frame = last_;
            age = std::chrono::duration<double>(clock::now() - last_at_).count();
        }

        double w = have_ ? std::clamp(frame.mode_pr() / 100.0, 0.0, 1.0) : 0.0;
        if (have_ && age > STALE_S) {
            // A silent client: ramp out from the last weight, then forget the frame.
            w *= std::max(0.0, 1.0 - (age - STALE_S) / RAMP_OUT_S);
            if (w == 0.0) {
                have_ = false;
                spdlog::warn("arm_sdk: no frame for {:.1f} s, joints handed back", age);
            }
        }

        // Joints the policy does not drive (the head) keep the command they had on the first cycle
        // (FixStand's hold), as the base to blend from and to return to.
        std::array<bool, 35> locked;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!held_set_) {
                for (int j : JOINTS) {
                    const auto& c = cmd.motor_cmd()[j];
                    const float q = override_[j] ? held_[j].q : c.q();
                    held_[j] = {q, c.kp(), c.kd()};
                }
                held_set_ = true;
            }
            base_ = held_;
            locked = locked_;
        }

        for (int j : JOINTS) {
            auto& c = cmd.motor_cmd()[j];
            const bool policy = std::find(driven.begin(), driven.end(), float(j)) != driven.end();
            // The policy rewrites its joints every cycle; elsewhere blend from the held command.
            const Cmd base = policy ? Cmd{c.q(), c.kp(), c.kd()} : base_[j];
            if (locked[j] && !policy) {
                c.q() = base.q; c.kp() = base.kp; c.kd() = base.kd;
                continue;
            }
            if (w <= 0.0) {
                if (!policy) { c.q() = base.q; c.kp() = base.kp; c.kd() = base.kd; }
                continue;
            }
            const auto& f = frame.motor_cmd()[j];
            c.q() = float((1 - w) * base.q + w * f.q());
            c.kp() = float((1 - w) * base.kp + w * f.kp());
            c.kd() = float((1 - w) * base.kd + w * f.kd());
            c.dq() = 0.0f;
            c.tau() = 0.0f;
        }
        weight_ = w;
    }

    // Whether arm_sdk holds the joints (any weight): the walking service reports 816 then.
    bool active() const { return weight_ > 0.0; }

    // A locked joint ignores the frames and keeps its held command (Perceptive's tilted head).
    void lock(int joint, bool locked) { std::lock_guard<std::mutex> l(mutex_); locked_[joint] = locked; }
    // The command a joint the policy does not drive is held at (and returns to).
    void hold(int joint, float q)
    {
        std::lock_guard<std::mutex> l(mutex_);
        held_[joint].q = q;
        override_[joint] = true;
    }

private:
    using clock = std::chrono::steady_clock;
    struct Cmd { float q = 0, kp = 0, kd = 0; };

    unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowCmd_> sub_;
    std::mutex mutex_;
    unitree_hg::msg::dds_::LowCmd_ frame_, last_;
    clock::time_point at_, last_at_;
    bool fresh_ = false, have_ = false, warned_ = false, held_set_ = false;
    std::array<Cmd, 35> held_{}, base_{};
    std::array<bool, 35> locked_{}, override_{};
    std::atomic<double> weight_{0.0};
};

// The controller's arm_sdk, or null when it is off.
inline std::shared_ptr<ArmSdk> arm_sdk;
