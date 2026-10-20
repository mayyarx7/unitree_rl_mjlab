import mujoco
import numpy as np
import onnxruntime as ort
import yaml

MODEL = "/home/aimat/mayyar/unitree_mujoco/unitree_robots/r1/scene.xml"
BASE = "deploy/robots/r1/config/policy/velocity/v0/"
POLICY = BASE + "exported/policy.onnx"
CONFIG = BASE + "params/deploy.yaml"

session = ort.InferenceSession(
    POLICY, providers=["CPUExecutionProvider"]
)
meta = session.get_modelmeta().custom_metadata_map

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

ids = np.array(cfg["joint_ids_map"], dtype=int)
kp = np.array(cfg["stiffness"], dtype=float)
kd = np.array(cfg["damping"], dtype=float)
default = np.array(cfg["default_joint_pos"], dtype=float)

old_scale = np.full(24, 0.25)
new_scale = np.array(
    [float(x) for x in meta["action_scale"].split(",")]
)

assert len(ids) == 24
assert len(kp) == len(kd) == len(default) == 24
assert len(new_scale) == 24
assert session.get_inputs()[0].shape == [1, 83]
assert session.get_outputs()[0].shape == [1, 24]

def rotate_inverse(q, v):
    w, x, y, z = q
    xyz = np.array([x, y, z])
    t = 2.0 * np.cross(-xyz, v)
    return v + w * t + np.cross(-xyz, t)

def run_test(name, scale):
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)

    qadr = np.array([
        m.jnt_qposadr[int(m.actuator_trnid[i, 0])]
        for i in ids
    ])
    vadr = np.array([
        m.jnt_dofadr[int(m.actuator_trnid[i, 0])]
        for i in ids
    ])

    torso = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
    )

    quat_sensor = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat"
    )
    gyro_sensor = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro"
    )

    assert quat_sensor >= 0 and gyro_sensor >= 0

    qa = int(m.sensor_adr[quat_sensor])
    ga = int(m.sensor_adr[gyro_sensor])

    mujoco.mj_forward(m, d)

    last_action = np.zeros(24)
    desired = default.copy()
    fall_time = None
    max_action = 0.0

    physics_dt = m.opt.timestep
    policy_dt = float(cfg["step_dt"])
    decimation = round(policy_dt / physics_dt)

    assert abs(decimation * physics_dt - policy_dt) < 1e-8

    print(f"\n=== {name} ===")
    print("Zeit | Torso_Z | COM_X | Pitch | MaxAction")

    for step in range(int(3.0 / physics_dt)):

        if step % decimation == 0:
            quat = d.sensordata[qa:qa+4].copy()
            gyro = d.sensordata[ga:ga+3].copy()

            gravity = rotate_inverse(
                quat, np.array([0.0, 0.0, -1.0])
            )

            # Neutraler Controller: Command und Phase = 0
            obs = np.concatenate([
                gyro,
                gravity,
                np.zeros(3),
                np.zeros(2),
                d.qpos[qadr] - default,
                d.qvel[vadr],
                last_action
            ]).astype(np.float32)

            assert obs.shape == (83,)

            action = session.run(
                None, {"obs": obs.reshape(1, 83)}
            )[0].reshape(-1).astype(float)

            if not np.isfinite(action).all():
                print("FEHLER: Policy liefert NaN/Inf")
                break

            last_action = action.copy()
            desired = default + scale * action
            max_action = max(
                max_action, float(np.max(np.abs(action)))
            )

        # Gleiche PD-Regelung in beiden Tests
        d.ctrl[:] = 0.0
        d.ctrl[ids] = (
            kp * (desired - d.qpos[qadr])
            - kd * d.qvel[vadr]
        )

        mujoco.mj_step(m, d)

        quat = d.qpos[3:7]
        w, x, y, z = quat

        pitch = np.degrees(np.arctan2(
            2 * (w*y + x*z),
            1 - 2 * (y*y + x*x)
        ))

        height = float(d.xpos[torso, 2])
        com = np.average(
            d.xipos, axis=0, weights=m.body_mass
        )

        if fall_time is None and (
            height < 0.35 or abs(pitch) > 60
        ):
            fall_time = d.time

        if step % round(0.5 / physics_dt) == 0:
            print(
                f"{d.time:4.2f} | "
                f"{height:7.3f} | "
                f"{com[0]:5.3f} | "
                f"{pitch:6.1f} | "
                f"{max_action:8.3f}"
            )

        if not np.isfinite(d.qpos).all():
            print("FEHLER: Simulation numerisch instabil")
            break

    print("\nErgebnis:", name)
    print("Sturzzeit:", (
        f"{fall_time:.3f}s"
        if fall_time is not None else
        "Kein Sturz innerhalb von 3 Sekunden"
    ))
    print("Letzte Torsohoehe:", round(height, 3))
    print("Letzter Pitch:", round(pitch, 1))
    print("Maximale rohe Aktion:", round(max_action, 3))

    return fall_time, height, pitch

print("Vergleich der Action-Scale-Werte")
print("Alte Scale:", old_scale)
print("Neue Scale:", new_scale)

a = run_test("A: ALTE SCALE 0.25", old_scale)
b = run_test("B: NEUE SCALE AUS ONNX", new_scale)

print("\n========== VERGLEICH ==========")
print("                ALT        NEU")
print(f"Torsohoehe:   {a[1]:8.3f}   {b[1]:8.3f}")
print(f"Pitch:        {a[2]:8.1f}   {b[2]:8.1f}")
print(
    "Sturzzeit:   ",
    "kein" if a[0] is None else round(a[0], 3),
    "     ",
    "kein" if b[0] is None else round(b[0], 3)
)
