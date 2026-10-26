# Unitree RL Mjlab

> [!IMPORTANT]
> **This is a fork** of [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)
> for the Unitree **R1**: our own velocity policy, run in low-level mode, teleoperated from a Meta Quest
> through [arena-r1-kit](https://github.com/MustafaOrhon/arena-r1-kit). The same controller binary runs
> against the simulator (`--network lo`) and the real robot (`--network <eth>`); only the DDS interface
> differs. The fork's changes are listed in [R1 fork](#-r1-fork); everything after it is the upstream README.

## 🤖 R1 fork

### What changed

**Controller (`deploy/robots/r1/r1_ctrl`)** – it now offers the interfaces Unitree's own R1 controller
offers, so arena-r1-kit drives our policy unchanged, in simulation and on the robot:

| Option | Effect |
|---|---|
| `--cmd-source joystick` | Default, as upstream: velocity from the remote's sticks. |
| `--cmd-source sport` | Serves the R1 walking service (`sport`, [`loco_server.h`](deploy/include/loco_server.h)): `SetVelocity` feeds the policy's velocity command, `GetFsmId` reports `811` (walking) in Velocity. Commands outside Velocity are refused; a command expires after its `duration`. The remote's sticks are ignored. |
| `--arm-sdk` | Takes upper-body targets on `rt/arm_sdk` ([`arm_sdk.h`](deploy/include/arm_sdk.h)) as Unitree's controller does: both arms, waist yaw and head (joints 13, 15–19, 22–26, 29–30), blended over the policy by `mode_pr` (0–100). Frames with the wrong `mode_machine` are dropped; after 0.5 s without a frame the weight ramps out over 1.5 s. While it holds the joints, `GetFsmId` reports `816` and `SetVelocity` is refused. |
| `--start-velocity` | **Simulation only.** Starts directly in Velocity with FixStand's gains and final pose, skipping the remote's `L2+Up` / `R1+X`. |
| `--lowstate-only` | Diagnostics: waits for `LowState` and prints tick / `mode_machine` every second, sends no motor commands. |

If `CYCLONEDDS_URI` is set, the controller leaves the interface to that XML instead of passing
`--network` to SDK2 (which would override it); see [`cyclonedds_local.xml`](cyclonedds_local.xml) and
[`docs/r1-dds-diagnose.md`](docs/r1-dds-diagnose.md) (German) for why LowState did not arrive on `lo`
under WSL2 and the fragmentation fix.

Other controller changes: `FixStand` in [`config.yaml`](deploy/robots/r1/config/config.yaml) also holds
the head (joints 29–30), the R1 `deploy.yaml` matches the trained policy, and the R1 build uses
`unitree_sdk2` from `/opt/unitree_robotics`.

**Simulator (`simulate/`)** – defaults to the R1 in a scene with a KUKA cell
(`src/assets/robots/unitree_r1/xmls/scene_kuka.xml`) and adds a simulated head camera
([`head_camera.h`](simulate/src/head_camera.h)): the head's views are rendered offscreen over EGL,
resampled into the real R1 fisheye ([`head_camera.json`](simulate/head_camera.json), same size and
intrinsics as on the robot) and published as H.264 over RTSP via ffmpeg at the address where the robot's
camera would be served, so arena-r1-kit's `stream.py` sees the same stream as on the robot.

| `simulate/config.yaml` key / flag | Meaning |
|---|---|
| `head_camera: 1` | Enable the head camera (needs mediamtx running with [`mediamtx_sim.yml`](simulate/mediamtx_sim.yml)). |
| `head_camera_model`, `head_camera_fps`, `head_camera_url` | Fisheye model, frame rate, RTSP publish address (default `rtsp://127.0.0.1:8554/robot`). |
| `ffmpeg` | Path of the ffmpeg binary (set to a local path; adjust for your machine). |
| `head_camera_stereo` / `--stereo 1` | Two eyes, stacked vertically (left on top) for depth in the Quest. Eye spacing is **assumed** 6.3 cm (`eye_*` cameras in `R1_C++.xml`) until the real R1 camera baseline is known. |
| `use_joystick` / `--joystick 0\|1` | Gamepad as the remote; off by default now, since the Quest sends the velocity. |

The viewer now creates its GL context over EGL, because a GLX context in the same process breaks the
head camera's EGL context on NVIDIA PRIME.

**Training** – R1 velocity runs and checkpoints are committed under `logs/rsl_rl/r1_velocity/`
(latest: `2026-10-20_14-19-20`, up to `model_14999.pt`). Train or resume with the upstream commands below
using task `Unitree-R1-Flat`.

**Helper scripts** (repo root, ad hoc): `check_r1_start_pose.py`, `compare_r1_scales.py`,
`test_r1_fall.py` (policy push test in plain MuJoCo), `patch_r1_start.py` (writes the policy's default
pose into the simulator's start pose), `ps5_receiver.py` (PS5 pad over UDP, superseded by the Quest).
Some contain absolute paths from the machine they were written on.

### Build

```bash
# unitree_sdk2 installed to /opt/unitree_robotics; mediamtx and ffmpeg available
cd deploy/robots/r1 && mkdir -p build && cd build && cmake .. && make -j
cd ../../../../simulate && mkdir -p build && cd build && cmake .. && make -j
```

### Run in simulation with the Quest

[`start_quest_sim.sh`](start_quest_sim.sh) starts everything in order and stops it with Ctrl+C. It
expects `arena-r1-kit` (with its `.venv`) next to this repo, mediamtx in `~/tools/mediamtx/` and ffmpeg
in `~/tools/ffmpeg/`; override with `ARENA=…`, `MEDIAMTX=…`, `FFMPEG_DIR=…`. Logs go to
`logs/quest_sim/<time>/`. `--stereo` needs an arena-r1-kit with `stream.py --stereo`.

```bash
./start_quest_sim.sh              # walking: left stick walks, right stick turns
./start_quest_sim.sh --upper      # plus head and right arm over rt/arm_sdk
./start_quest_sim.sh --stereo     # stereo head camera (combinable with --upper)
```

What it runs, if you want to do it by hand:

```bash
mediamtx simulate/mediamtx_sim.yml                                   # 1. video relay
cd deploy/robots/r1/build && ./r1_ctrl --network lo \
    --cmd-source sport --arm-sdk --start-velocity                    # 2. controller
cd simulate/build && ./unitree_mujoco --joystick 0 [--stereo 1]      # 3. simulator + head camera
cd arena-r1-kit && .venv/bin/python stream.py --http --webrtc --walk \
    --iface lo --source rtsp://127.0.0.1:8554/robot [--stereo] \
    [--head --arm --teleop-scale 0.7 --arm-slew 40]                  # 4. Quest page on :8080
adb reverse tcp:8080 tcp:8080 && adb reverse tcp:8189 tcp:8189      # 5. Quest over USB
```

Then open `http://localhost:8080` in the Quest browser and enter VR (`/flat` shows the video on the laptop).

### Run on the real R1

Same controller, without `--start-velocity`, on the robot's Ethernet interface:

```bash
./r1_ctrl --network <eth> --cmd-source sport --arm-sdk
```

Enter FixStand (`L2+Up`) and Velocity (`R1+X`) on the remote as usual, then point `stream.py` at the
robot's interface and camera (`rtsp://192.168.123.161:8551/video`). The real R1 has two head cameras;
whether they form a stereo pair usable for `--stereo` is still open.

---


## ✳️ Overview
Unitree RL Mjlab is a reinforcement learning project built upon the
[mjlab](https://github.com/mujocolab/mjlab.git), using MuJoCo as its 
physics simulation backend, currently supporting Unitree Go2, A2, As2, G1, R1, H1_2 and H2.

Mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s proven API
with best-in-class [MuJoCo](https://github.com/google-deepmind/mujoco_warp)
physics to provide lightweight, modular abstractions for RL robotics research
and sim-to-real deployment.

<div align="center">

| <div align="center">  MuJoCo </div>                                                                                                                                           | <div align="center"> Physical </div>                                                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/g1-velocity.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> | <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/g1-velocity-real.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> |

</div>


## 📦 Installation and Configuration

Please refer to [setup.md](doc/setup_en.md) for installation and configuration steps.


## 🔁 Process Overview

The basic workflow for using reinforcement learning to achieve motion control is:

`Train` → `Play` → `Sim2Real`

- **Train**: The agent interacts with the MuJoCo simulation and optimizes policies through reward maximization.
- **Play**: Replay trained policies to verify expected behavior.
- **Sim2Real**: Deploy trained policies to physical Unitree robots for real-world execution.


## 🛠️ Usage Guide

### 1. Velocity Tracking Training

Run the following command to train a velocity tracking policy:

```bash
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

Multi-GPU Training: Scale to multiple GPUs using --gpu-ids:

```bash
python scripts/train.py Unitree-G1-Flat \
  --gpu-ids 0 1 \
  --env.scene.num-envs=4096
```

- The first argument (e.g., Mjlab-Velocity-Flat-Unitree-G1) specifies the training task.
Available velocity tracking tasks:
  - Unitree-Go2-Flat
  - Unitree-G1-Flat
  - Unitree-G1-23Dof-Flat
  - Unitree-H1_2-Flat
  - Unitree-A2-Flat
  - Unitree-R1-Flat

> [!NOTE]
> For more details, refer to the mjlab documentation:
> [mjlab documentation](https://mujocolab.github.io/mjlab/index.html).

### 2. Motion Imitation Training

Train a Unitree G1 to mimic reference motion sequences.

<div style="margin-left: 20px;">

#### 2.1 Prepare Motion Files

Prepare csv motion files in mjlab/motions/g1/ and convert them to npz format:

```bash
python scripts/csv_to_npz.py \
--input-file src/assets/motions/g1/dance1_subject2.csv \
--output-name dance1_subject2.npz \
--input-fps 30 \
--output-fps 50 \
--robot g1 # g1 or g1_23dof
```

**npz files will be stored at:**：`src/motions/g1/...`

#### 2.2 Training

After generating the NPZ file, launch imitation training:

```bash
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --env.scene.num-envs=4096
```

Available tasks:
  - Unitree-G1-Tracking-No-State-Estimation
  - Unitree-G1-23Dof-Tracking-No-State-Estimation

</div>

> [!NOTE]
> For detailed motion imitation instructions, refer to the BeyondMimic documentation:
> [BeyondMimic documentation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup).

#### ⚙️  Parameter Description
- `--env.scene`: simulation scene configuration (e.g., num_envs, dt, ground type, gravity, disturbances)
- `--env.observations`: observation space configuration (e.g., joint state, IMU, commands, etc.)
- `--env.rewards`: reward terms used for policy optimization
- `--env.commands`: task commands (e.g., velocity, pose, or motion targets)
- `--env.terminations`: termination conditions for each episode
- `--agent.seed`: random seed for reproducibility
- `--agent.resume`: resume from the last saved checkpoint when enabled
- `--agent.policy`: policy network architecture configuration
- `--agent.algorithm`: reinforcement learning algorithm configuration (PPO, hyperparameters, etc.)

**Training results are stored at**：`logs/rsl_rl/<robot>_(velocity | tracking)/<date_time>/model_<iteration>.pt`

### 3. Simulation Validation

To visualize policy behavior in MuJoCo:

Velocity tracking:
```bash
python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
```

Motion imitation:
```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --checkpoint_file=logs/rsl_rl/g1_tracking/2026-xx-xx_xx-xx-xx/model_xx.pt
```

**Note**：

- During training, policy.onnx and policy.onnx.data are also exported for deployment onto physical robots.

**Visualization**：

| Go2                              | G1                             | H1_2                               | G1_mimic                          |
|----------------------------------|--------------------------------|------------------------------------|-----------------------------------|
| ![go2](doc/gif/go2-velocity.gif) | ![g1](doc/gif/g1-velocity.gif) | ![h1_2](doc/gif/h1_2-velocity.gif) | ![g1_mimic](doc/gif/g1-mimic.gif) |

### 4. Real Deployment

Before deployment, install the required communication tools:
- [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds.git)
- [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2.git)

<div style="margin-left: 20px;">

#### 4.1 Power On the Robot
Start the robot in suspended state and wait until it enters `zero-torque` mode.

#### 4.2 Enable Debug Mode
While in `zero-torque` mode, press `L2 + R2` on the controller. The robot will enter `debug mode` with joint damping enabled.

#### 4.3 Connect to the Robot
Connect your PC to the robot via Ethernet. Configure the network as:
- Address：`192.168.123.222`
- Netmask：`255.255.255.0`

Use `ifconfig` to determine the Ethernet device name for deployment.

#### 4.4 Compilation

Example: Unitree G1 velocity control.
Place `policy.onnx` and `policy.onnx.data` into: `deploy/robots/g1/config/policy/velocity/v0/exported`.
Then compile:

```bash
cd deploy/robots/g1
mkdir build && cd build
cmake .. && make
```

#### 4.5 Deployment

## 4.5.1 Simulation Deployment

Before deploying on the real robot, it is recommended to perform simulation deployment using [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
to prevent abnormal behaviors on the physical robot. This framework has already integrated it.

Build unitree_mujoco：

```bash
cd simulate
mkdir build && cd build
cmake .. && make -j8
```

Launch the simulator (note that a gamepad must be connected):

```bash
./simulate/build/unitree_mujoco
```

You can select the corresponding robot in `simulate/config`

Launch the simulation control program:

```bash
cd deploy/robots/g1/build
./g1_ctrl --network=lo
```

## 4.5.2 Real-Robot Deployment

Launch the control program on the real robot:

```bash
cd deploy/robots/g1/build
./g1_ctrl --network=enp5s0
```

**Arguments**：
- `network`: The network interface used to connect to the robot. Use `lo` for simulation deployment, and `enp5s0` for the real robot(You can check it using the `ifconfig` command) 

</div>

**Deployment Results**：

| Go2                                                    | G1                                                    | H1_2           | G1_mimic                                           |
|--------------------------------------------------------|-------------------------------------------------------|----------------|----------------------------------------------------|
| <img src="doc/gif/go2-velocity-real.gif" width="300"/> | <img src="doc/gif/g1-velocity-real.gif" width="300"/> | <img src="doc/gif/h1_2-velocity-real.gif" width="300"/> | <img src="doc/gif/g1-mimic-real.gif" width="300"/> |


## 🎉  Acknowledgements

This project would not be possible without the contributions of the following repositories:

- [mjlab](https://github.com/mujocolab/mjlab.git): training and execution framework
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking.git): versatile humanoid motion tracking framework
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): reinforcement learning algorithm implementation
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git): GPU-accelerated rendering and simulation interface
- [mujoco](https://github.com/google-deepmind/mujoco.git): high-fidelity rigid-body physics engine
