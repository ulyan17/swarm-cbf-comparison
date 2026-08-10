"""Контроллер CBF с QP, slack и отслеживанием насыщения управления."""
import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple
from src.config import CBFConfig, SimConfig
from src.environment import Environment, Agent
from src.controllers.base import BaseController

class CBFController(BaseController):
    def __init__(self, cbf_cfg: CBFConfig, sim_cfg: SimConfig):
        self.cfg = cbf_cfg
        self.sim_cfg = sim_cfg
        self.last_slack = 0.0
        self.last_saturated = []   # список флагов насыщения для агентов

    def compute(self, env: Environment) -> list[np.ndarray]:
        actions = []
        self.last_saturated = []
        for ag in env.agents:
            u_nom = -self.cfg.kp * (ag.state[:2] - ag.goal) - self.cfg.kd * ag.state[2:4]
            constraints = self._build_constraints(ag, env)
            res = self._solve_qp(u_nom, constraints)
            u_opt = res.x[:2] if res.success else u_nom
            # Клиппинг по u_max
            u_clipped = np.clip(u_opt, -self.sim_cfg.u_max, self.sim_cfg.u_max)
            self.last_saturated.append(not np.allclose(u_opt, u_clipped, atol=1e-6))
            actions.append(u_clipped)
        return actions

    def _build_constraints(self, ag: Agent, env: Environment) -> List[Tuple[np.ndarray, float]]:
        cons = []
        pos, vel = ag.state[:2], ag.state[2:4]
        # препятствия
        for obs in env.obstacles:
            d_min = self.sim_cfg.safe_dist + obs.radius
            d_vec = pos - obs.pos
            d = np.linalg.norm(d_vec)
            if d < 1e-6: d = 1e-6
            unit = d_vec / d
            h = d - d_min
            dot_h = np.dot(unit, vel)
            psi1 = dot_h + self.cfg.kappa1 * h
            v_norm2 = np.dot(vel, vel)
            term = v_norm2 / d - dot_h**2 / d
            A = unit
            b = -(term + self.cfg.kappa1 * dot_h + self.cfg.kappa2 * psi1)
            cons.append((A, b))
        # другие агенты
        for other in env.agents:
            if ag.id == other.id: continue
            d_vec = pos - other.state[:2]
            d = np.linalg.norm(d_vec)
            if d < 1e-6: d = 1e-6
            unit = d_vec / d
            rel_vel = vel - other.state[2:4]
            h = d - self.sim_cfg.safe_dist
            dot_h = np.dot(unit, rel_vel)
            psi1 = dot_h + self.cfg.kappa1 * h
            rel_v_norm2 = np.dot(rel_vel, rel_vel)
            term = rel_v_norm2 / d - dot_h**2 / d
            A = unit
            b = -(term + self.cfg.kappa1 * dot_h + self.cfg.kappa2 * psi1)
            cons.append((A, b))
        return cons

    def _solve_qp(self, u_nom: np.ndarray, constraints: List[Tuple[np.ndarray, float]]):
        def objective(x):
            u = x[:2]
            s = x[2]
            return 0.5 * np.sum((u - u_nom)**2) + self.cfg.slack_weight * s**2
        cons = []
        for A, b in constraints:
            cons.append({'type': 'ineq', 'fun': lambda x, A=A, b=b: np.dot(A, x[:2]) - b + x[2]})
        cons.append({'type': 'ineq', 'fun': lambda x: x[2]})
        x0 = np.array([u_nom[0], u_nom[1], 0.0])
        res = minimize(objective, x0, method='SLSQP', constraints=cons,
                       options={'maxiter': 100, 'disp': False})
        # Используем порог 1e-3, чтобы не считать числовой шум за нарушение
        self.last_slack = res.x[2] if res.success else 1.0
        return res