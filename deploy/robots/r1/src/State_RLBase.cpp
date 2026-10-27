#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "loco_server.h"
#include "arm_sdk.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "heightmap_input.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

namespace
{
// Perceptive holds the head tilted down for the depth camera (config: head_pitch); the policy
// does not drive the head. The next policy state without a head_pitch hands it back.
constexpr int HEAD_PITCH = 29, HEAD_YAW = 30;
constexpr double HEAD_RAMP_S = 1.0;
struct HeadHold { bool active = false; float restore = 0.0f; };
HeadHold head_hold;
}


State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    if (loco_server) {
        env->robot->data.velocity_command = [](float& vx, float& vy, float& wz) {
            loco_server->command(vx, vy, wz);
        };
    }

    if (cfg["head_pitch"]) head_pitch = cfg["head_pitch"].as<float>();

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }

    // run() comes every FSM cycle (1 ms) while the state is current: a gap means it was just entered.
    const auto now = std::chrono::steady_clock::now();
    const bool entered = now - last_run_ > std::chrono::milliseconds(50);
    last_run_ = now;
    auto& head = lowcmd->msg_.motor_cmd()[HEAD_PITCH];
    if (head_pitch) {
        if (entered) {
            entered_ = now;
            head_from_ = head.q();
            if (!head_hold.active) head_hold = {true, head.q()};
            if (arm_sdk) { arm_sdk->lock(HEAD_PITCH, true); arm_sdk->lock(HEAD_YAW, true); }
            spdlog::info("{}: head tilting to {:.2f} rad for the depth camera", getStateString(), *head_pitch);
        }
        const double s = std::min(1.0, std::chrono::duration<double>(now - entered_).count() / HEAD_RAMP_S);
        head.q() = float(head_from_ + s * (*head_pitch - head_from_));
        if (arm_sdk) arm_sdk->hold(HEAD_PITCH, head.q());
    } else if (head_hold.active) {
        head.q() = head_hold.restore;
        if (arm_sdk) {
            arm_sdk->hold(HEAD_PITCH, head_hold.restore);
            arm_sdk->lock(HEAD_PITCH, false);
            arm_sdk->lock(HEAD_YAW, false);
        }
        head_hold.active = false;
        spdlog::info("{}: head back to {:.2f} rad", getStateString(), head_hold.restore);
    }

    if (arm_sdk) {
        arm_sdk->apply(lowcmd->msg_, env->robot->data.joint_ids_map, lowstate->msg_.mode_machine());
    }
}