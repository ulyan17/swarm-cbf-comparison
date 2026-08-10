"""Демонстрация с выбором контроллера и масштаба роя."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse
from src.config import SimConfig, CBFConfig, CrystalConfig, GNNConfig
from src.environment import Environment
from src.controllers.cbf import CBFController
from src.controllers.crystal import CrystalController
from src.controllers.gcbf import GCBFController

def run_demo(controller_name: str = 'CBF', num_agents: int = 5,
             num_obstacles: int = 2, max_steps: int = 500, seed: int = 42):
    sim_cfg = SimConfig()
    env = Environment(sim_cfg)

    np.random.seed(seed)
    for _ in range(num_agents):
        pos = np.random.rand(2) * (sim_cfg.world_size - 2) + 1
        vel = (np.random.rand(2) - 0.5) * 0.3
        goal = np.random.rand(2) * (sim_cfg.world_size - 2) + 1
        env.add_agent(pos, vel, goal)
    for _ in range(num_obstacles):
        pos = np.random.rand(2) * (sim_cfg.world_size - 2) + 1
        env.add_obstacle(pos[0], pos[1], sim_cfg.obs_radius)

    # Выбор контроллера
    if controller_name == 'CBF':
        ctrl = CBFController(CBFConfig(), sim_cfg)
    elif controller_name == 'Crystal':
        ctrl = CrystalController(CrystalConfig(), sim_cfg)
    elif controller_name == 'GCBF+':
        try:
            ctrl = GCBFController(GNNConfig(), sim_cfg, model_path='gcbf_model.pt')
        except FileNotFoundError:
            print("Модель gcbf_model.pt не найдена. Сначала запустите train_gcbf.py")
            return
    else:
        raise ValueError("Выберите 'CBF', 'Crystal' или 'GCBF+'")

    # Визуализация
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, sim_cfg.world_size)
    ax.set_ylim(0, sim_cfg.world_size)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Препятствия
    for obs in env.obstacles:
        circle = Circle(obs.pos, obs.radius, color='gray', alpha=0.7)
        ax.add_patch(circle)

    # Цели
    for ag in env.agents:
        ax.plot(ag.goal[0], ag.goal[1], 'gx', markersize=10)

    # Точки агентов
    agent_points = []
    for ag in env.agents:
        point, = ax.plot(ag.state[0], ag.state[1], 'bo', markersize=6)
        agent_points.append(point)

    plt.title(f'Контроллер: {controller_name} | Агентов: {num_agents} | Шаг 0/{max_steps}')
    plt.ion()
    plt.show()

    for step in range(1, max_steps + 1):
        actions = ctrl.compute(env)
        for ag, u in zip(env.agents, actions):
            ag.update(u, sim_cfg)
        for i, ag in enumerate(env.agents):
            agent_points[i].set_data([ag.state[0]], [ag.state[1]])
        plt.title(f'Контроллер: {controller_name} | Агентов: {num_agents} | Шаг {step}/{max_steps}')
        plt.pause(0.001)
        if env.all_goals_reached():
            print("Все цели достигнуты!")
            break

    plt.ioff()
    plt.show()
    print(f"Завершено за {step} шагов. Столкновений: {env.collision_pairs()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', default='CBF', choices=['CBF','Crystal','GCBF+'])
    parser.add_argument('--agents', type=int, default=5)
    parser.add_argument('--obstacles', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_demo(args.controller, args.agents, args.obstacles, seed=args.seed)