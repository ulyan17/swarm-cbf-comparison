"""Глобальные параметры симуляции и алгоритмов."""
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SimConfig:
    dt: float = 0.05
    max_vel: float = 1.5
    safe_dist: float = 0.5
    agent_radius: float = 0.15
    obs_radius: float = 0.3
    goal_tolerance: float = 0.3
    world_size: float = 10.0
    max_steps: int = 600
    u_max: float = 2.0          # максимальное ускорение, м/с²

@dataclass
class CBFConfig:
    kappa1: float = 2.0
    kappa2: float = 2.0
    slack_weight: float = 100.0
    kp: float = 1.0
    kd: float = 0.5

@dataclass
class CrystalConfig:
    k_att: float = 1.0
    k_rep: float = 0.5
    d_eq: float = 0.8
    k_damp: float = 0.6
    k_obs_rep: float = 0.8

@dataclass
class GNNConfig:
    r_sensor: float = 3.0
    hidden_dim: int = 64
    lr: float = 1e-3
    batch_size: int = 128
    num_epochs: int = 30
    num_train_scenes: int = 2000
    norm_mean: Optional[List[float]] = None
    norm_std: Optional[List[float]] = None
    edge_mean: Optional[List[float]] = None
    edge_std: Optional[List[float]] = None

class NodeType:
    AGENT_SELF = 0
    AGENT = 1
    GOAL = 2
    OBS = 3