// rt/lowstate into shared memory for the height map process (../heightmap_node.py).
//
// Python's DDS library decodes every message while holding the GIL, and at the simulator's
// 1000 messages a second that starved the mapping. Here the SDK decodes in C++ and each sample
// with a new tick goes into a ring buffer the Python side reads as one numpy block.
//
//   ./build/lowstate_shm                      # simulator: DDS on lo, /dev/shm/r1_lowstate
//   ./build/lowstate_shm --iface eth0         # robot: the cable's interface
//
// Layout (r1_perception/lowstate_source.py reads it): LowstateShmHeader, then capacity entries.
// The writer fills entry count % capacity, sets its seq to count + 1 last, then raises count;
// a reader takes an entry only if its seq is the one it expects (otherwise it was overwritten).

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

constexpr uint32_t NUM_MOTORS = 31;  // the R1's motors in rt/lowstate (of the message's 35 slots)
constexpr uint32_t CAPACITY = 1024;  // about 2 s at 500 Hz

struct LowstateShmHeader
{
    char magic[8];                 // "R1LOWST1"
    uint32_t header_size;
    uint32_t entry_size;
    uint32_t capacity;
    uint32_t num_motors;
    std::atomic<uint64_t> count;   // entries written so far
};

struct LowstateEntry
{
    std::atomic<uint64_t> seq;     // index + 1 once the entry is complete
    uint32_t tick;                 // rt/lowstate tick (ms)
    uint32_t reserved;
    double wall;                   // arrival, CLOCK_REALTIME seconds (as the depth frames' stamps)
    float q[NUM_MOTORS];
    float quat[4];                 // IMU, w x y z
    float pad;
};
static_assert(sizeof(LowstateShmHeader) == 32, "header layout");
static_assert(sizeof(LowstateEntry) == 168, "entry layout");

static std::atomic<bool> stop{false};

int main(int argc, char** argv)
{
    std::string iface = "lo", name = "/r1_lowstate";
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        if (key == "--iface") iface = argv[i + 1];
        else if (key == "--shm") name = argv[i + 1];
        else { std::cerr << "usage: lowstate_shm [--iface lo] [--shm /r1_lowstate]\n"; return 2; }
    }

    const size_t bytes = sizeof(LowstateShmHeader) + CAPACITY * sizeof(LowstateEntry);
    const int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0644);
    if (fd < 0 || ftruncate(fd, off_t(bytes)) != 0) { std::cerr << "cannot create /dev/shm" << name << "\n"; return 1; }
    void* mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mem == MAP_FAILED) { std::cerr << "cannot map /dev/shm" << name << "\n"; return 1; }
    std::memset(mem, 0, bytes);
    auto* header = static_cast<LowstateShmHeader*>(mem);
    auto* entries = reinterpret_cast<LowstateEntry*>(static_cast<char*>(mem) + sizeof(LowstateShmHeader));
    std::memcpy(header->magic, "R1LOWST1", 8);
    header->header_size = sizeof(LowstateShmHeader);
    header->entry_size = sizeof(LowstateEntry);
    header->capacity = CAPACITY;
    header->num_motors = NUM_MOTORS;

    unitree::robot::ChannelFactory::Instance()->Init(0, iface);
    uint32_t last_tick = 0;
    bool have_tick = false;
    unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_> sub("rt/lowstate");
    sub.InitChannel([&](const void* msg) {
        const auto& s = *static_cast<const unitree_hg::msg::dds_::LowState_*>(msg);
        if (have_tick && s.tick() == last_tick) return;  // published again without a new step
        last_tick = s.tick();
        have_tick = true;
        const uint64_t n = header->count.load(std::memory_order_relaxed);
        LowstateEntry& e = entries[n % CAPACITY];
        e.seq.store(0, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        e.tick = s.tick();
        e.wall = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        for (uint32_t i = 0; i < NUM_MOTORS; ++i) e.q[i] = s.motor_state()[i].q();
        for (int i = 0; i < 4; ++i) e.quat[i] = s.imu_state().quaternion()[i];
        e.seq.store(n + 1, std::memory_order_release);
        header->count.store(n + 1, std::memory_order_release);
    }, 0);

    std::signal(SIGINT, [](int) { stop = true; });
    std::signal(SIGTERM, [](int) { stop = true; });
    std::cout << "[lowstate_shm] rt/lowstate on " << iface << " -> /dev/shm" << name << " (" << CAPACITY
              << " entries of " << sizeof(LowstateEntry) << " bytes)" << std::endl;
    uint64_t reported = 0;
    while (!stop) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        const uint64_t n = header->count.load();
        std::cout << "[lowstate_shm] " << (n - reported) / 5.0 << " samples/s" << std::endl;
        reported = n;
    }
    munmap(mem, bytes);
    close(fd);
    shm_unlink(name.c_str());
    std::cout << "[lowstate_shm] stopped" << std::endl;
    return 0;
}
