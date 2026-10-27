#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "loco_server.h"
#include "arm_sdk.h"
#include "heightmap_input.h"
#include <cstdlib>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    param::po::options_description extra("R1 diagnostics");
    extra.add_options()
        ("lowstate-only", "Receive LowState without publishing motor commands or starting the FSM")
        ("start-velocity", "SIMULATION ONLY: start directly in Velocity (as --start-state Velocity)")
        ("start-state", param::po::value<std::string>(),
         "SIMULATION ONLY: start directly in this policy state (Velocity, Perceptive)")
        ("cmd-source", param::po::value<std::string>()->default_value("joystick"),
         "velocity command: joystick (the remote's sticks) or sport (the walking service, SetVelocity)")
        ("arm-sdk", "take upper-body commands on rt/arm_sdk in Velocity, blended by their weight");
    auto vm = param::helper(argc, argv, extra);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     R1 Controller \n";

    // Unitree DDS Config
    // An explicit interface makes SDK2 supply its own XML, overriding CYCLONEDDS_URI.
    const char* dds_uri = std::getenv("CYCLONEDDS_URI");
    unitree::robot::ChannelFactory::Instance()->Init(
        0, dds_uri && *dds_uri ? "" : vm["network"].as<std::string>());

    if (vm.count("lowstate-only")) {
        FSMState::lowstate = std::make_shared<LowState_t>();
        spdlog::info("Waiting for connection to robot...");
        FSMState::lowstate->wait_for_connection();
        spdlog::info("Connected to robot.");
        uint32_t previous_tick = 0;
        while (true) {
            sleep(1);
            std::lock_guard<std::mutex> lock(FSMState::lowstate->mutex_);
            spdlog::info("LowState: tick={}, mode_machine={}",
                         FSMState::lowstate->msg_.tick(), FSMState::lowstate->msg_.mode_machine());
            if (FSMState::lowstate->msg_.tick() == previous_tick) {
                spdlog::warn("LowState tick has not advanced (simulation paused or reception stalled).");
            }
            previous_tick = FSMState::lowstate->msg_.tick();
        }
    }

    const auto cmd_source = vm["cmd-source"].as<std::string>();
    if (cmd_source != "joystick" && cmd_source != "sport") {
        spdlog::critical("--cmd-source must be joystick or sport, got {}", cmd_source);
        exit(-1);
    }
    // Made before the FSM, so the Velocity state takes its commands from it.
    if (cmd_source == "sport") loco_server = std::make_shared<LocoServer>();

    init_fsm_state();
    if (vm.count("arm-sdk")) {
        arm_sdk = std::make_shared<ArmSdk>();
        spdlog::info("Upper body: rt/arm_sdk blended in Velocity (arms, waist yaw, head).");
    }

    FSMState::lowcmd->msg_.mode_machine() = 1;
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    std::string start_state = vm.count("start-state") ? vm["start-state"].as<std::string>() : "";
    if (start_state.empty() && vm.count("start-velocity")) start_state = "Velocity";
    if (!start_state.empty())
    {
        spdlog::warn("SIMULATION TEST: Starting directly in {}", start_state);
        // As if FixStand had run: its gains and final pose hold every joint the policy does not drive (the head).
        auto fix = param::config["FSM"]["FixStand"];
        auto kp = fix["kp"].as<std::vector<float>>();
        auto kd = fix["kd"].as<std::vector<float>>();
        auto q = fix["qs"].as<std::vector<std::vector<float>>>().back();
        for (size_t i = 0; i < kp.size(); ++i) {
            auto& motor = FSMState::lowcmd->msg_.motor_cmd()[i];
            motor.kp() = kp[i];
            motor.kd() = kd[i];
            motor.q() = q[i];
            motor.dq() = motor.tau() = 0;
        }
        fsm->start(start_state);
    }
    else
    {
        fsm->start();
    }

    if (loco_server) {
        const int velocity_id = FSMStringMap.right.at("Velocity");
        const int perceptive_id = FSMStringMap.right.count("Perceptive") ? FSMStringMap.right.at("Perceptive") : -1;
        CtrlFSM* f = fsm.get();
        loco_server->set_fsm_id([f, velocity_id, perceptive_id] {
            const int id = f->currentStateId();
            if (id != velocity_id && id != perceptive_id) return id;
            // As on Unitree's controller: 816 while arm_sdk holds the upper body, and no walking then.
            return arm_sdk && arm_sdk->active() ? LocoServer::FSM_ARM_SDK : LocoServer::FSM_WALKING;
        });
        loco_server->set_accepting([f, velocity_id, perceptive_id] {
            const int id = f->currentStateId();
            if (arm_sdk && arm_sdk->active()) return false;
            if (id == velocity_id) return true;
            // Perceptive walks only on a fresh height map; without one the robot stands.
            return id == perceptive_id && heightmap_input && !heightmap_input->falling_back();
        });
        loco_server->Init();
        loco_server->Start(false);
        spdlog::info("Velocity commands from the walking service ({}); the remote's sticks are ignored.",
                     unitree::robot::r1::LOCO_SERVICE_NAME);
    }

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R1 + X] to start controlling the robot.\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

