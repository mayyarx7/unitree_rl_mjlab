import mujoco
import numpy as np
import onnxruntime as ort
import yaml

MODEL = "/home/aimat/mayyar/unitree_mujoco/unitree_robots/r1/scene.xml"
BASE = "deploy/robots/r1/config/policy/velocity/v0"
POLICY = BASE + "/exported/policy.onnx"
CONFIG = BASE + "/params/deploy.yaml"

DURATION = 10.0
PUSH_START = 3.0
PUSH_DURATION = 0.15

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

session = ort.InferenceSession(
    POLICY, providers=["CPUExecutionProvider"]
)

meta = session.get_modelmeta().custom_metadata_map
scale = np.array(
    [float(v) for v in meta["action_scale"].split(",")]
)

ids = np.array(cfg["joint_ids_map"], dtype=int)
kp = np.array(cfg["stiffness"], dtype=float)
kd = np.array(cfg["damping"], dtype=float)
default = np.array(cfg["default_joint_pos"], dtype=float)

assert len(ids) == len(kp) == len(kd) == len(default) == len(scale) == 24
assert session.get_inputs()[0].shape == [1, 83]

def gravity_in_body(q):
    w, x, y, z = q
    v = np.array([0.0, 0.0, -1.0])
    xyz = -np.array([x, y, z])
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)

def run_test(force):
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)

    torso = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
    )

    qadr = np.array([
        m.jnt_qposadr[int(m.actuator_trnid[i, 0])]
        for i in ids
    ])

    vadr = np.array([
        m.jnt_dofadr[int(m.actuator_trnid[i, 0])]
        for i in ids
    ])

    quat_sensor = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat"
    )
    gyro_sensor = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro"
    )

    qa = int(m.sensor_adr[quat_sensor])
    ga = int(m.sensor_adr[gyro_sensor])

    mujoco.mj_forward(m, d)

    action = np.zeros(24)
    desired = default.copy()

    decimation = round(cfg["step_dt"] / m.opt.timestep)
    assert abs(decimation * m.opt.timestep - cfg["step_dt"]) < 1e-8

    fall_time = None
    min_height = float("inf")
    max_tilt = 0.0

    print("\n" + "=" * 55)
    print(f"TEST: Stoßkraft {force} N")
    print("=" * 55)
    print("Zeit    Torsohöhe    Neigung    Status")

    steps = int(DURATION / m.opt.timestep)
    print_every = max(1, round(1.0 / m.opt.timestep))

    for step in range(steps):

        if step % decimation == 0:
            quat = d.sensordata[qa:qa + 4].copy()
            gyro = d.sensordata[ga:ga + 3].copy()

            obs = np.concatenate([
                gyro,
                gravity_in_body(quat),
                np.zeros(3),      # Joystick neutral
                np.zeros(2),      # Gangphase bei Stillstand
                d.qpos[qadr] - default,
                d.qvel[vadr],
                action,
            ]).astype(np.float32)

            action = session.run(
                None, {"obs": obs.reshape(1, 83)}
            )[0].reshape(24).astype(float)

            if not np.isfinite(action).all():
                print("ABBRUCH: Ungültige Policy-Aktion")
                return None

            desired = default + scale * action

        # PD-Motorsteuerung
        d.ctrl[:] = 0.0
        d.ctrl[ids] = (
            kp * (desired - d.qpos[qadr])
            - kd * d.qvel[vadr]
        )

        # Externer horizontaler Stoß auf den Torso
        d.xfrc_applied[:] = 0.0

        if PUSH_START <= d.time < PUSH_START + PUSH_DURATION:
            d.xfrc_applied[torso, 0] = force

        mujoco.mj_step(m, d)

        if not np.isfinite(d.qpos).all():
            print("ABBRUCH: Simulation numerisch instabil")
            return None

        height = float(d.xpos[torso, 2])

        # Neigung der freien Basis gegen die Vertikale
        quat_base = d.qpos[3:7]
        w, x, y, z = quat_base

        vertical_alignment = 1.0 - 2.0 * (x*x + y*y)
        tilt = np.degrees(
            np.arccos(np.clip(vertical_alignment, -1.0, 1.0))
        )

        min_height = min(min_height, height)
        max_tilt = max(max_tilt, tilt)

        fallen = height < 0.35 or tilt > 60.0

        if fallen and fall_time is None:
            fall_time = float(d.time)

        if step % print_every == 0:
            status = "GESTUERZT" if fallen else "OK"
            print(
                f"{d.time:5.2f}s  "
                f"{height:8.3f}m  "
                f"{tilt:8.1f}°  {status}"
            )

    print("\nERGEBNIS")
    print("Minimale Torsohöhe:", round(min_height, 3), "m")
    print("Maximale Neigung:", round(max_tilt, 1), "Grad")

    if fall_time is None:
        print("KEIN STURZ innerhalb von 10 Sekunden")
    else:
        print(f"STURZSCHWELLE bei {fall_time:.3f} Sekunden überschritten")

    return fall_time


print("R1 STABILITAETSTEST")
print("Policy:", POLICY)
print("Trainingslauf:", meta.get("run_path", "unbekannt"))
print("Stoß bei t=3s für 0.15s")
print("Der Test verändert keine Projektdateien.")

results = {}

for force in [0, 100, 250, 500]:
    results[force] = run_test(force)

print("\n" + "=" * 55)
print("GESAMTAUSWERTUNG")
print("=" * 55)

for force, fall_time in results.items():
    if fall_time is None:
        print(f"{force:4d} N : Kein Sturz erkannt")
    else:
        print(f"{force:4d} N : Sturzschwelle bei {fall_time:.3f}s")
