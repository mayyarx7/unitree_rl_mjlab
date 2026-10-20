from pathlib import Path
import re
import shutil
import yaml

source = Path("simulate/src/main.cc")
config = Path(
    "deploy/robots/r1/config/policy/velocity/v0/params/deploy.yaml"
)

cfg = yaml.safe_load(config.read_text())
ids = [int(x) for x in cfg["joint_ids_map"]]
positions = cfg["default_joint_pos"]

assert len(ids) == len(positions) == 24

code = source.read_text()
marker = "void SetR1StartPose(const mjModel* model, mjData* data)"

if marker in code:
    raise SystemExit("Patch ist bereits vorhanden. Keine Änderung.")

# Die Funktion wird vor PhysicsLoop definiert.
match = re.search(r"\bvoid\s+PhysicsLoop\s*\(", code)
if not match:
    raise SystemExit("PhysicsLoop nicht gefunden. Keine Änderung.")

id_values = ", ".join(map(str, ids))
pos_values = ", ".join(f"{float(x):.9g}" for x in positions)

helper = f"""
// Initialpose fuer Unitree R1 entsprechend der Velocity-Policy.
void SetR1StartPose(const mjModel* model, mjData* data)
{{
  if (!model || !data) return;

  // Andere Robotermodelle nicht veraendern.
  if (mj_name2id(model, mjOBJ_BODY, "torso_link") < 0 ||
      model->nu != 29 || model->nq < 7) return;

  const int actuator_ids[24] = {{{id_values}}};
  const mjtNum positions[24] = {{{pos_values}}};

  // Erst alle Indizes validieren.
  for (int i = 0; i < 24; ++i) {{
    int actuator_id = actuator_ids[i];
    if (actuator_id < 0 || actuator_id >= model->nu) return;
    int joint_id = model->actuator_trnid[2 * actuator_id];
    if (joint_id < 0 || joint_id >= model->njnt) return;
    if (model->jnt_qposadr[joint_id] < 0 ||
        model->jnt_qposadr[joint_id] >= model->nq) return;
  }}

  data->qpos[2] = 0.74;
  data->qpos[3] = 1.0;
  data->qpos[4] = 0.0;
  data->qpos[5] = 0.0;
  data->qpos[6] = 0.0;

  for (int i = 0; i < 24; ++i) {{
    int joint_id = model->actuator_trnid[2 * actuator_ids[i]];
    int qpos_id = model->jnt_qposadr[joint_id];
    data->qpos[qpos_id] = positions[i];
  }}

  mju_zero(data->qvel, model->nv);
  mju_zero(data->ctrl, model->nu);
}}

"""

code = code[:match.start()] + helper + code[match.start():]

# Initialisierung beim Start
old_start = """sim->Load(m, d, filename);
      mj_forward(m, d);"""
new_start = """sim->Load(m, d, filename);
      SetR1StartPose(m, d);
      mj_forward(m, d);"""

# Beide Wege zum Nachladen von Modellen
old_reload = """m = mnew;
          d = dnew;
          mj_forward(m, d);"""
new_reload = """m = mnew;
          d = dnew;
          SetR1StartPose(m, d);
          mj_forward(m, d);"""

# Reset mit Backspace
old_reset = """mj_resetData(m, d);
      mj_forward(m, d);"""
new_reset = """mj_resetData(m, d);
      SetR1StartPose(m, d);
      mj_forward(m, d);"""

expected = [
    (old_start, new_start, 1),
    (old_reload, new_reload, 2),
    (old_reset, new_reset, 1),
]

for old, new, count in expected:
    actual = code.count(old)
    if actual != count:
        raise SystemExit(
            f"Codeabschnitt unerwartet: {actual} statt {count}. "
            "Keine Datei veraendert."
        )
    code = code.replace(old, new)

backup = Path("simulate/src/main.cc.before_r1_pose")
shutil.copy2(source, backup)
source.write_text(code)

print("R1-Startpose eingebaut.")
print("Backup:", backup)
print("24 Gelenke aus deploy.yaml uebernommen.")
print("Initialisierung, Nachladen und Reset angepasst.")
