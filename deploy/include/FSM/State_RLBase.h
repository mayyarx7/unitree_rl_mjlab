// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include <chrono>
#include <optional>
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        for (size_t i = 0; i < env->robot->data.joint_ids_map.size(); ++i)
        {
            const int motor_id =
                static_cast<int>(env->robot->data.joint_ids_map[i]);

            auto& cmd = lowcmd->msg_.motor_cmd()[motor_id];

            cmd.kp() = env->robot->data.joint_stiffness[i];
            cmd.kd() = env->robot->data.joint_damping[i];
            cmd.dq() = 0.0f;
            cmd.tau() = 0.0f;
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;

    // Robot-specific use (the R1's Perceptive holds its head tilted): a head pitch from the
    // state's config, and when run() last ran, to tell a fresh entry into the state.
    std::optional<float> head_pitch;
    std::chrono::steady_clock::time_point last_run_{}, entered_{};
    float head_from_ = 0.0f;
};

REGISTER_FSM(State_RLBase)
