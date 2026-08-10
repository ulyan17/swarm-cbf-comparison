"""Проведение серии экспериментов для сравнения контроллеров."""
import numpy as np
import random
from typing import Dict, List
from src.config import SimConfig, CBFConfig, CrystalConfig, GNNConfig
from src.environment import Environment
from src.controllers.cbf import CBFController
from src.controllers.crystal import CrystalController
from src.controllers.gcbf import GCBFController
from src.simulation import Simulation

def run_experiments(seeds: int = 10, agent_counts: List[int] = None,
                    use_gcbf: bool = True) -> Dict[str, List[Dict]]:
    """
    Запускает тестовые сценарии для трёх контроллеров.
    Возвращает словарь: results[ctrl_name] = список словарей метрик.
    """
    if agent_counts is None:
        agent_counts = [3, 5, 10]

    sim_cfg = SimConfig()
    cbf_cfg = CBFConfig()
    crystal_cfg = CrystalConfig()
    gnn_cfg = GNNConfig()

    controllers = {
        'CBF': CBFController(cbf_cfg, sim_cfg),
        'Crystal': CrystalController(crystal_cfg, sim_cfg),
    }
    if use_gcbf:
        try:
            controllers['GCBF+'] = GCBFController(gnn_cfg, sim_cfg, model_path='gcbf_model.pt')
        except FileNotFoundError:
            print("Модель gcbf_model.pt не найдена. GCBF+ пропущен.")
            use_gcbf = False

    results = {name: [] for name in controllers}

    for N in agent_counts:
        for seed in range(seeds):
            np.random.seed(seed)
            random.seed(seed)

            # Создаём базовую среду
            env_base = Environment(sim_cfg)
            for _ in range(N):
                pos = np.random.rand(2) * (sim_cfg.world_size - 2) + 1
                vel = (np.random.rand(2) - 0.5) * 0.3
                goal = np.random.rand(2) * (sim_cfg.world_size - 2) + 1
                env_base.add_agent(pos, vel, goal)
            # Добавляем пару препятствий
            env_base.add_obstacle(sim_cfg.world_size * 0.3, sim_cfg.world_size * 0.3, sim_cfg.obs_radius)
            env_base.add_obstacle(sim_cfg.world_size * 0.7, sim_cfg.world_size * 0.7, sim_cfg.obs_radius)

            for name, ctrl in controllers.items():
                # Копируем среду
                env = Environment(sim_cfg)
                for ag in env_base.agents:
                    env.add_agent(ag.state[:2].copy(), ag.state[2:4].copy(), ag.goal.copy())
                for obs in env_base.obstacles:
                    env.add_obstacle(obs.pos[0], obs.pos[1], obs.radius)

                sim = Simulation(env, ctrl, max_steps=sim_cfg.max_steps)
                metrics = sim.run()
                results[name].append(metrics)

    return results

def print_summary(results: Dict[str, List[Dict]]):
    """Выводит сводную статистику по метрикам."""
    for name, metrics_list in results.items():
        steps = [m['steps'] for m in metrics_list]
        collisions = [m['collisions'] for m in metrics_list]
        min_dists = [m['min_dist'] for m in metrics_list]
        avg_accs = [m['avg_acc'] for m in metrics_list]
        times = [m['time_total'] for m in metrics_list]

        print(f"\n{'='*50}")
        print(f"Контроллер: {name}")
        print(f"  Среднее время симуляции: {np.mean(times):.2f} с")
        print(f"  Среднее число шагов:     {np.mean(steps):.1f}")
        print(f"  Суммарные столкновения:  {np.sum(collisions)}")
        print(f"  Среднее мин. расстояние: {np.mean(min_dists):.3f}")
        print(f"  Среднее ускорение:       {np.mean(avg_accs):.3f}")

if __name__ == '__main__':
    # Для быстрой проверки без обученной модели GCBF+
    res = run_experiments(seeds=3, agent_counts=[3, 5], use_gcbf=False)
    print_summary(res)