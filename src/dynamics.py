"""Динамика двойного интегратора для точечных агентов."""
import numpy as np
from src.config import SimConfig

def double_integrator(state: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Правая часть ОДУ:
    state = [px, py, vx, vy]^T
    u = [ax, ay]^T
    Возвращает производную состояния.
    """
    px, py, vx, vy = state
    return np.array([vx, vy, u[0], u[1]])

def euler_step(state: np.ndarray, u: np.ndarray, config: SimConfig) -> np.ndarray:
    """Один шаг явного метода Эйлера с ограничением скорости."""
    new_state = state + double_integrator(state, u) * config.dt
    v = np.linalg.norm(new_state[2:4])
    if v > config.max_vel:
        new_state[2:4] = new_state[2:4] / v * config.max_vel
    return new_state