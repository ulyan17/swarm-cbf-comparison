"""Кристаллический контроллер."""
import numpy as np
from src.config import CrystalConfig, SimConfig
from src.environment import Environment
from src.controllers.base import BaseController

class CrystalController(BaseController):
    def __init__(self, crystal_cfg: CrystalConfig, sim_cfg: SimConfig):
        self.cfg = crystal_cfg
        self.sim_cfg = sim_cfg

    def compute(self, env: Environment) -> list[np.ndarray]:
        actions = []
        for i, ag in enumerate(env.agents):
            pos = ag.state[:2]
            vel = ag.state[2:4]
            u = self.cfg.k_att * (ag.goal - pos)
            for j, other in enumerate(env.agents):
                if i == j: continue
                d_vec = pos - other.state[:2]
                d = np.linalg.norm(d_vec)
                if d < 1e-4: d = 1e-4
                factor = (1.0 - self.cfg.d_eq / d) / (d**3)
                u += self.cfg.k_rep * d_vec * factor
            for obs in env.obstacles:
                d_vec = pos - obs.pos
                d = np.linalg.norm(d_vec)
                if d < 1e-4: d = 1e-4
                d_eff = max(d - obs.radius, 0.01)
                factor = (1.0 - self.sim_cfg.safe_dist / d_eff) / (d_eff**3)
                u += self.cfg.k_obs_rep * d_vec * factor
            u -= self.cfg.k_damp * vel
            u = np.clip(u, -self.sim_cfg.u_max, self.sim_cfg.u_max)
            actions.append(u)
        return actions