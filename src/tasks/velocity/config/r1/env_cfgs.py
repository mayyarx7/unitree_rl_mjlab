"""Unitree R1 velocity environment configurations."""

from dataclasses import replace

import mjlab.terrains as terrain_gen
from src.assets.robots import (
  R1_ACTION_SCALE,
  get_r1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import ConstantNoiseCfg, NoiseModelWithAdditiveBiasCfg, UniformNoiseCfg
from src.tasks.velocity import mdp as r1_mdp
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# Steps up to 20 cm: an EUR pallet is 14.4 cm high, the rest is margin. The "pallet" terrains
# have 0.8 m deep steps, a pallet's 800 mm side; "stairs" 0.3 m deep ones. Pyramids start the
# robot on top (steps down), inverted ones in the pit (steps up).
R1_STEPS_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.15),
    "stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.15,
      step_height_range=(0.02, 0.20),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.15,
      step_height_range=(0.02, 0.20),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "pallet": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.2,
      step_height_range=(0.02, 0.20),
      step_width=0.8,
      platform_width=1.2,
      border_width=1.0,
    ),
    "pallet_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.2,
      step_height_range=(0.02, 0.20),
      step_width=0.8,
      platform_width=1.2,
      border_width=1.0,
    ),
    "rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.15,
      noise_range=(0.01, 0.05),
      noise_step=0.01,
      border_width=0.25,
    ),
  },
  add_lights=True,
)


def unitree_r1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree R1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_r1_robot_cfg()}

  # Set raycast sensor frame to R1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = R1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.05

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_r1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree R1 flat terrain velocity configuration."""
  cfg = unitree_r1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def unitree_r1_steps_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree R1 blind step-up configuration (EUR pallet, stairs up to 20 cm).

  The actor observes exactly what the flat policy does (no height scan), so the exported policy
  drops into deploy/robots/r1's Velocity state unchanged. The critic keeps the height scan: it
  only exists in training and makes the steps easier to learn. Start the actor from the flat
  policy with train.py's --init-actor-from.
  """
  cfg = unitree_r1_rough_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = replace(R1_STEPS_TERRAINS_CFG, curriculum=not play)
  cfg.scene.terrain.max_init_terrain_level = 2
  if play:
    cfg.scene.terrain.terrain_generator.num_cols = 6
    cfg.scene.terrain.terrain_generator.num_rows = 5
    cfg.scene.terrain.terrain_generator.border_width = 10.0

  # Blind actor: the deployed observation set.
  del cfg.observations["actor"].terms["height_scan"]

  # Rays see only the terrain (geom group 0), not the robot's own meshes (r1.xml: visuals in
  # group 2): standing, 4 of the scan's 187 rays hit the hips or feet, which no camera on the
  # robot would report as ground.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.include_geom_groups = (0,)

  # Ground under each foot, for the clearance reward.
  foot_ground = tuple(
    RayCastSensorCfg(
      name=f"{side}_foot_ground",
      frame=ObjRef(type="body", name=f"{side}_ankle_roll_link", entity="robot"),
      ray_alignment="yaw",
      pattern=GridPatternCfg(size=(0.2, 0.1), resolution=0.1),
      max_distance=1.0,
      exclude_parent_body=True,
      include_geom_groups=(0,),
    )
    for side in ("left", "right")
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + foot_ground

  site_names = ("left_foot", "right_foot")
  # Swing feet 12 cm above the ground under them (flat: 10 cm above z = 0).
  cfg.rewards["foot_clearance"] = RewardTermCfg(
    func=r1_mdp.feet_clearance_terrain,
    weight=-1.0,
    params={
      "target_height": 0.12,
      "sensor_names": tuple(s.name for s in foot_ground),
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    },
  )
  cfg.rewards["foot_stumble"] = RewardTermCfg(
    func=r1_mdp.feet_stumble,
    weight=-1.0,
    params={"sensor_name": "feet_ground_contact"},
  )

  # Walking speeds for steps: no later stage up to 2 m/s.
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
    ]
  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def unitree_r1_perceptive_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree R1 perceptive step-up configuration (EUR pallet, stairs up to 20 cm).

  As Unitree-R1-Steps, but the actor also gets the height scan the way the perception chain
  (depth camera -> height map) will deliver it: 10 Hz updates, 60-160 ms late, noisy, with
  filled holes and an offset per episode. The critic keeps the exact scan.
  """
  cfg = unitree_r1_steps_env_cfg(play=play)

  terrain_scan = next(s for s in cfg.scene.sensors if s.name == "terrain_scan")
  assert isinstance(terrain_scan, RayCastSensorCfg)
  cfg.observations["actor"].terms["height_scan"] = ObservationTermCfg(
    func=r1_mdp.height_scan_perceived,
    params={
      "sensor_name": "terrain_scan",
      "update_period": 5,  # policy steps of 20 ms: a new map at 10 Hz, the depth camera's rate
      "noise_std": 0.025,
      "hole_prob": 0.1,
    },
    # The whole map shifted up or down for an episode (camera pose, odometry).
    noise=NoiseModelWithAdditiveBiasCfg(
      noise_cfg=ConstantNoiseCfg(bias=0.0),
      bias_noise_cfg=UniformNoiseCfg(n_min=-0.03, n_max=0.03),
      sample_bias_per_component=False,
    ),
    scale=1 / terrain_scan.max_distance,
    delay_min_lag=3,  # 60-160 ms at 50 Hz
    delay_max_lag=8,
    delay_update_period=5,
    delay_hold_prob=0.5,
  )

  # It sees the edges, so it need not lift every step high (Steps: 12 cm). Only a foot below
  # 8 cm costs: a pallet's 14.4 cm need the foot higher, which the symmetric cost punished.
  cfg.rewards["foot_clearance"].params["target_height"] = 0.08
  cfg.rewards["foot_clearance"].params["only_below"] = True

  if not play:
    # Fine-tuning for the pallet: steps of 8-20 cm in every row (from 2-20 cm), so most
    # robots train near 14.4 cm. Training from scratch wants the 2 cm rows back.
    generator = cfg.scene.terrain.terrain_generator
    generator.sub_terrains = {
      name: replace(sub, step_height_range=(0.08, 0.20))
      if hasattr(sub, "step_height_range") else sub
      for name, sub in generator.sub_terrains.items()
    }
    # Progress after 3 m: out of a pallet pyramid from its centre (0.6 + 3 x 0.8 m), not half
    # the 8 m sub-terrain. With random commands few robots made 4 m in 20 s.
    cfg.curriculum["terrain_levels"].params["move_up_distance"] = 3.0

  return cfg
