// The R1's depth image in simulation: renders a pinhole depth view from the head (the "depth_camera"
// in the MJCF) and publishes it in shared memory for the perception process, which turns it into
// the height map the perceptive policy walks on. The R1 EDU's camera module delivers depth at
// 544 x 448 and 10 Hz; how it does that on the robot is not known yet, so this layout is the sim's
// own and a reader on the robot gets an adapter.
//
// Shared memory /dev/shm/<name>: DepthShmHeader, then width * height uint16 depths in millimetres
// along the optical axis, rows from the top, 0 where nothing was measured. A frame is consistent
// when seq is even and the same before and after copying it (seqlock).

#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <mujoco/mujoco.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

struct DepthCameraConfig
{
    bool enabled = false;
    std::string camera = "depth_camera";  // MJCF camera: the image's pose and vertical field of view
    int width = 544;
    int height = 448;
    int fps = 10;
    float max_range = 10.0f;               // metres; farther counts as not measured
    std::string shm_name = "/r1_depth";
};

struct DepthShmHeader
{
    char magic[8];         // "R1DEPTH1"
    uint32_t header_size;  // bytes before the depths
    uint32_t width, height;
    float fx, fy, cx, cy;  // pinhole intrinsics, OpenCV convention (pixel centres at integers)
    float depth_unit;      // metres per depth count
    uint32_t reserved;
    std::atomic<uint64_t> seq;  // odd while a frame is being written
    double stamp;          // capture time, CLOCK_REALTIME seconds (as the reader stamps rt/lowstate)
    double sim_time;       // mjData time at capture
};

class DepthCamera
{
public:
    explicit DepthCamera(DepthCameraConfig cfg) : cfg_(std::move(cfg)) {}

    ~DepthCamera()
    {
        if (has_scene_) mjv_freeScene(&scn_);
        if (shm_) munmap(shm_, shm_bytes_);
        if (fd_ >= 0) { close(fd_); shm_unlink(cfg_.shm_name.c_str()); }
    }

    // For a (new) model, with the renderer's context current and its offscreen buffer set.
    bool build(const mjModel* m)
    {
        camid_ = mj_name2id(m, mjOBJ_CAMERA, cfg_.camera.c_str());
        if (camid_ < 0) {
            std::cerr << "[depth camera] camera " << cfg_.camera << " not in the model; depth off\n";
            return false;
        }
        if (m->vis.global.offwidth < cfg_.width || m->vis.global.offheight < cfg_.height) {
            std::cerr << "[depth camera] the model's offscreen buffer is smaller than " << cfg_.width << "x"
                      << cfg_.height << "; depth off\n";
            return false;
        }
        if (has_scene_) mjv_freeScene(&scn_);
        mjv_defaultScene(&scn_);
        mjv_makeScene(m, &scn_, 10000);
        // Only geometry: no shadows, no floor reflection (it draws a mirrored scene), no sky.
        scn_.flags[mjRND_SHADOW] = 0;
        scn_.flags[mjRND_REFLECTION] = 0;
        scn_.flags[mjRND_SKYBOX] = 0;
        has_scene_ = true;
        mjv_defaultCamera(&cam_);
        cam_.type = mjCAMERA_FIXED;
        cam_.fixedcamid = camid_;

        // MuJoCo's projection: vertical field of view fovy, square pixels, centred.
        fy_ = 0.5 * cfg_.height / std::tan(m->cam_fovy[camid_] * M_PI / 360.0);
        const double extent = m->stat.extent;
        znear_ = m->vis.map.znear * extent;
        zfar_ = m->vis.map.zfar * extent;
        raw_.resize(size_t(cfg_.width) * cfg_.height);
        if (!open_shm()) return false;
        write_header();
        period_ = std::chrono::microseconds(1000000 / cfg_.fps);
        std::cout << "[depth camera] " << cfg_.width << "x" << cfg_.height << " at " << cfg_.fps << " Hz, "
                  << "horizontal " << 2 * std::atan(0.5 * cfg_.width / fy_) * 180 / M_PI << " x vertical "
                  << m->cam_fovy[camid_] << " deg, /dev/shm" << cfg_.shm_name << std::endl;
        return true;
    }

    // Whether a frame is due now; if so, the caller updates the scene (under the model lock) and
    // then calls render().
    bool due()
    {
        const auto now = std::chrono::steady_clock::now();
        if (now < next_) return false;
        next_ = std::max(next_ + period_, now);
        return true;
    }

    void update_scene(const mjModel* m, mjData* d, const mjvOption* opt)
    {
        mjv_updateScene(m, d, opt, nullptr, &cam_, mjCAT_ALL, &scn_);
        sim_time_ = d->time;
        stamp_ = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    }

    void render(mjrContext* con)
    {
        const mjrRect vp = {0, 0, cfg_.width, cfg_.height};
        mjr_render(vp, &scn_, con);
        mjr_readPixels(nullptr, raw_.data(), vp, con);

        auto* h = static_cast<DepthShmHeader*>(shm_);
        uint16_t* out = reinterpret_cast<uint16_t*>(static_cast<char*>(shm_) + sizeof(DepthShmHeader));
        const uint64_t s = h->seq.load(std::memory_order_relaxed);
        h->seq.store(s + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        // Depth buffer to metres along the optical axis (as MuJoCo's own renderer converts it);
        // readPixels rows run bottom to top.
        const float max_mm = cfg_.max_range * 1000.0f;
        for (int r = 0; r < cfg_.height; ++r) {
            const float* src = raw_.data() + size_t(cfg_.height - 1 - r) * cfg_.width;
            uint16_t* dst = out + size_t(r) * cfg_.width;
            for (int c = 0; c < cfg_.width; ++c) {
                const double z = znear_ / (1.0 - src[c] * (1.0 - znear_ / zfar_));
                const float mm = float(z * 1000.0);
                dst[c] = (src[c] >= 1.0f || mm > max_mm) ? 0 : uint16_t(std::lround(mm));
            }
        }
        h->stamp = stamp_;
        h->sim_time = sim_time_;
        std::atomic_thread_fence(std::memory_order_release);
        h->seq.store(s + 2, std::memory_order_release);
    }

private:
    bool open_shm()
    {
        if (shm_) return true;
        shm_bytes_ = sizeof(DepthShmHeader) + size_t(cfg_.width) * cfg_.height * sizeof(uint16_t);
        fd_ = shm_open(cfg_.shm_name.c_str(), O_CREAT | O_RDWR, 0644);
        if (fd_ < 0 || ftruncate(fd_, off_t(shm_bytes_)) != 0) {
            std::cerr << "[depth camera] cannot create /dev/shm" << cfg_.shm_name << "; depth off\n";
            return false;
        }
        shm_ = mmap(nullptr, shm_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (shm_ == MAP_FAILED) {
            shm_ = nullptr;
            std::cerr << "[depth camera] cannot map /dev/shm" << cfg_.shm_name << "; depth off\n";
            return false;
        }
        std::memset(shm_, 0, sizeof(DepthShmHeader));
        return true;
    }

    // The header's fixed part, for the current model's camera.
    void write_header()
    {
        auto* h = static_cast<DepthShmHeader*>(shm_);
        std::memcpy(h->magic, "R1DEPTH1", 8);
        h->header_size = sizeof(DepthShmHeader);
        h->width = cfg_.width;
        h->height = cfg_.height;
        h->fx = h->fy = float(fy_);
        h->cx = 0.5f * (cfg_.width - 1);
        h->cy = 0.5f * (cfg_.height - 1);
        h->depth_unit = 0.001f;
    }

    DepthCameraConfig cfg_;
    int camid_ = -1;
    mjvCamera cam_;
    mjvScene scn_;
    bool has_scene_ = false;
    double fy_ = 0, znear_ = 0, zfar_ = 0;
    std::vector<float> raw_;
    double sim_time_ = 0, stamp_ = 0;
    std::chrono::steady_clock::duration period_{};
    std::chrono::steady_clock::time_point next_{};
    int fd_ = -1;
    void* shm_ = nullptr;
    size_t shm_bytes_ = 0;
};
