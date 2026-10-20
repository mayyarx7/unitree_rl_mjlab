#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
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
        ("start-velocity", "SIMULATION ONLY: start directly in Velocity");
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

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 1;
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    if (vm.count("start-velocity"))
    {
        spdlog::warn("SIMULATION TEST: Starting directly in Velocity");
        fsm->start("Velocity");
    }
    else
    {
        fsm->start();
    }

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R1 + X] to start controlling the robot.\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

