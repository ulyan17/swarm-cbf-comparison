"""Компоненты среды моделирования: агенты, препятствия, среда."""
import numpy as np
from typing import List, Optional, Tuple
from src.config import SimConfig
from src.dynamics import euler_step

class Obstacle:
    """Статическое препятствие."""
    def __init__(self, x: float, y: float, radius: float = 0.3):
        self.pos = np.array([x, y])
        self.radius = radius

class Agent:
    """Агент роя."""
    def __init__(self, idx: int, pos: np.ndarray, vel: np.ndarray, goal: np.ndarray):
        self.id = idx
        self.state = np.concatenate([pos, vel])   # [px, py, vx, vy]
        self.goal = np.array(goal)
        self.history: List[np.ndarray] = []

    def update(self, u: np.ndarray, config: SimConfig) -> None:
        self.state = euler_step(self.state, u, config)
        self.history.append(self.state.copy())

class Environment:
    """Среда, содержащая агентов и препятствия."""
    def __init__(self, config: SimConfig):
        self.config = config
        self.agents: List[Agent] = []
        self.obstacles: List[Obstacle] = []

    def add_agent(self, pos: Tuple[float, float], vel: Tuple[float, float],
                  goal: Tuple[float, float]) -> None:
        idx = len(self.agents)
        agent = Agent(idx, np.array(pos), np.array(vel), np.array(goal))
        self.agents.append(agent)

    def add_obstacle(self, x: float, y: float, radius: Optional[float] = None) -> None:
        r = radius if radius is not None else self.config.obs_radius
        self.obstacles.append(Obstacle(x, y, r))

    def step(self, controller) -> None:
        """Один шаг симуляции: вычисляем управления и обновляем агентов."""
        actions = controller.compute(self)
        for ag, u in zip(self.agents, actions):
            ag.update(u, self.config)

    def all_goals_reached(self) -> bool:
        return all(np.linalg.norm(ag.state[:2] - ag.goal) < self.config.goal_tolerance
                   for ag in self.agents)

    def collision_pairs(self) -> int:
        """Количество нарушений безопасного расстояния (агент-агент и агент-препятствие)."""
        count = 0
        for i in range(len(self.agents)):
            for j in range(i+1, len(self.agents)):
                d = np.linalg.norm(self.agents[i].state[:2] - self.agents[j].state[:2])
                if d < self.config.safe_dist:
                    count += 1
        for ag in self.agents:
            for obs in self.obstacles:
                d = np.linalg.norm(ag.state[:2] - obs.pos) - obs.radius
                if d < self.config.agent_radius:
                    count += 1
        return count

    def min_safety_distance(self) -> float:
        """Минимальное расстояние до соседа или препятствия."""
        min_d = float('inf')
        for i in range(len(self.agents)):
            for j in range(i+1, len(self.agents)):
                d = np.linalg.norm(self.agents[i].state[:2] - self.agents[j].state[:2])
                min_d = min(min_d, d)
        for ag in self.agents:
            for obs in self.obstacles:
                d = np.linalg.norm(ag.state[:2] - obs.pos) - obs.radius
                min_d = min(min_d, d)
        return min_d if np.isfinite(min_d) else 0.0