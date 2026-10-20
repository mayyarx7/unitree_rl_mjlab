#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/dds_wrapper/common/crc.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

// Passive, bounded diagnostic: no LowCmd writer, FSM or policy.
int main(int argc, char** argv)
{
    using Message = unitree_hg::msg::dds_::LowState_;
    const char* uri = std::getenv("CYCLONEDDS_URI");
    unitree::robot::ChannelFactory::Instance()->Init(
        0, uri && *uri ? "" : (argc > 1 ? argv[1] : "lo"));
    std::atomic<unsigned> received{0}, invalid{0};
    std::atomic<uint32_t> tick{0};
    auto channel = unitree::robot::ChannelFactory::Instance()->CreateRecvChannel<Message>(
        "rt/lowstate", [&](const void* data) {
            const auto& msg = *static_cast<const Message*>(data);
            std::array<uint32_t, sizeof(Message) / sizeof(uint32_t)> words{};
            std::memcpy(words.data(), &msg, sizeof(Message));
            bool valid = crc32_core(words.data(), words.size() - 1) == msg.crc();
            for (const auto& motor : msg.motor_state()) {
                valid &= std::isfinite(motor.q()) && std::isfinite(motor.dq()) &&
                         std::isfinite(motor.tau_est());
            }
            float norm = 0;
            for (float value : msg.imu_state().quaternion()) norm += value * value;
            valid &= std::isfinite(norm) && std::abs(norm - 1.0f) < 0.01f;
            for (float value : msg.imu_state().gyroscope()) valid &= std::isfinite(value);
            for (float value : msg.imu_state().accelerometer()) valid &= std::isfinite(value);
            if (!valid) ++invalid;
            tick = msg.tick();
            ++received;
        });
    unsigned previous = 0;
    unsigned active_intervals = 0;
    for (int i = 0; i < 10; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        auto reader = channel->GetReader()->GetNative();
        unsigned count = received.load();
        if (count > previous) ++active_intervals;
        previous = count;
        std::cout << "received=" << count << " invalid=" << invalid.load()
                  << " tick=" << tick.load()
                  << " matched=" << reader.subscription_matched_status().current_count()
                  << " incompatible_qos=" << reader.requested_incompatible_qos_status().total_count()
                  << std::endl;
    }
    channel.reset(); // Stop callbacks before destroying their captured counters.
    return active_intervals >= 5 && invalid.load() == 0 ? 0 : 1;
}
