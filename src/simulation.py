"""Класс Simulation: запуск одного эпизода и сбор метрик."""
import time
import numpy as np
from src.environment import Environment
from src.controllers.base import BaseController

class Simulation:
    def __init__(self, env: Environment, controller: BaseController, max_steps: int = 600):
        self.env = env
        self.controller = controller
        self.max_steps = max_steps
        self.metrics = {}

    def run(self) -> dict:
        """Запускает симуляцию и возвращает метрики."""
        start_time = time.time()
        step_count = 0
        for step in range(self.max_steps):
            step_count += 1
            self.env.step(self.controller)
            if self.env.all_goals_reached():
                break
        elapsed = time.time() - start_time

        # Собираем метрики
        collisions = self.env.collision_pairs()
        min_dist = self.env.min_safety_distance()
        # Среднее ускорение (оценка по изменению скорости на последнем шаге)
        avg_acc = 0.0
        if self.env.agents:
            acc_sum = 0.0
            for ag in self.env.agents:
                if len(ag.history) > 1:
                    dv = ag.history[-1][2:4] - ag.history[-2][2:4]
                    acc_sum += np.linalg.norm(dv) / self.env.config.dt
            avg_acc = acc_sum / len(self.env.agents)

        self.metrics = {
            'steps': step_count,
            'time_total': elapsed,
            'collisions': collisions,
            'min_dist': min_dist,
            'avg_acc': avg_acc
        }
        return self.metrics