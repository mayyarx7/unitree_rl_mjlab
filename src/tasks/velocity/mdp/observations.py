from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.mdp.observations import height_scan
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase



class height_scan_perceived:
  """Height scan as the perception chain delivers it, for policies that walk on what they see.

  The chain (depth camera -> height map, see deploy) publishes a new map only every
  update_period steps (10 Hz at the 50 Hz policy rate), so between updates the policy sees the
  last map. Each new map gets per-cell Gaussian noise and holes; a hole is filled with the
  median of the map, as the chain fills cells it could not measure. A constant offset per
  episode and the chain's latency come from the term's noise and delay settings.

  Values as height_scan: frame height minus ground height per ray, in metres.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    period = cfg.params["update_period"]
    self.held: torch.Tensor | None = None
    # Envs update at different steps, as unsynchronised cameras would.
    self.phase = torch.randint(0, period, (env.num_envs,), device=env.device)
    self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    # A reset robot gets a new map at once instead of the one from before the reset.
    self.fresh[slice(None) if env_ids is None else env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    update_period: int,
    noise_std: float,
    hole_prob: float,
  ) -> torch.Tensor:
    heights = height_scan(env, sensor_name)  # [B, N]
    new_map = heights + noise_std * torch.randn_like(heights)
    holes = torch.rand_like(heights) < hole_prob
    median = new_map.median(dim=1, keepdim=True).values
    new_map = torch.where(holes, median, new_map)

    if self.held is None:
      self.held = new_map.clone()
    update = self.fresh | ((env.episode_length_buf + self.phase) % update_period == 0)
    self.held = torch.where(update.unsqueeze(1), new_map, self.held)
    self.fresh[:] = False
    return self.held.clone()
