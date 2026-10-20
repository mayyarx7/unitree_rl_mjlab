import mujoco
import numpy as np
import yaml

model_path = "/home/aimat/mayyar/unitree_mujoco/unitree_robots/r1/scene.xml"
config_path = "deploy/robots/r1/config/policy/velocity/v0/params/deploy.yaml"

m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

with open(config_path) as f:
    cfg = yaml.safe_load(f)

ids = cfg["joint_ids_map"]
default = cfg["default_joint_pos"]

for actuator_id, pos in zip(ids, default):
    joint_id = int(m.actuator_trnid[actuator_id, 0])
    qpos_id = int(m.jnt_qposadr[joint_id])
    d.qpos[qpos_id] = pos

mujoco.mj_forward(m, d)

torso_id = mujoco.mj_name2id(
    m, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
)

print("Basishöhe:", d.qpos[2])
print("Torsohöhe:", d.xpos[torso_id, 2])
print("Orientierung:", d.qpos[3:7])
print("Gelenke entsprechend der Policy initialisiert.")
print("Keine Simulationsdateien verändert.")
