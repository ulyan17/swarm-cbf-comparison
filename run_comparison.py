"""Сбор расширенных метрик для сравнения CBF, Crystal, GCBF+."""
import sys
sys.path.insert(0, '.')
import os
import csv
import time
import numpy as np
import random
from src.config import SimConfig, CBFConfig, CrystalConfig, GNNConfig
from src.environment import Environment
from src.controllers.cbf import CBFController
from src.controllers.crystal import CrystalController
from src.controllers.gcbf import GCBFController

sim_cfg = SimConfig()
cbf_cfg = CBFConfig()
crystal_cfg = CrystalConfig()
gnn_cfg = GNNConfig()
gnn_cfg.hidden_dim = 128

# Загрузка статистик нормализации
stats_data = np.load('experiments/50k/training_data.npz', allow_pickle=True)
gnn_cfg.norm_mean = stats_data['norm_mean'].tolist()
gnn_cfg.norm_std  = stats_data['norm_std'].tolist()
gnn_cfg.edge_mean = stats_data['edge_mean'].tolist()
gnn_cfg.edge_std  = stats_data['edge_std'].tolist()

GCBF_MODEL_PATH = 'experiments/50k/gcbf_model.pt'

N_list = [5, 10, 20]
num_seeds = 30
max_steps = 600
output_csv = 'comparison_results.csv'
trajectories_dir = 'trajectories'
os.makedirs(trajectories_dir, exist_ok=True)

representative_seeds = {5: [0, 5, 10], 10: [0, 5, 10], 20: [0, 5, 10]}

def measure_time(controller, env, warmup=5, trials=20):
    env_copy = Environment(env.config)
    for ag in env.agents:
        env_copy.add_agent(ag.state[:2].copy(), ag.state[2:4].copy(), ag.goal.copy())
    for obs in env.obstacles:
        env_copy.add_obstacle(obs.pos[0], obs.pos[1], obs.radius)

    for _ in range(warmup):
        actions = controller.compute(env_copy)
        for ag, u in zip(env_copy.agents, actions):
            ag.update(u, env_copy.config)

    times = []
    for _ in range(trials):
        start = time.perf_counter()
        actions = controller.compute(env_copy)
        for ag, u in zip(env_copy.agents, actions):
            ag.update(u, env_copy.config)
        end = time.perf_counter()
        times.append(end - start)
    return np.median(times) * 1000

def generate_scene(N, seed):
    sim_cfg_local = SimConfig()
    rng = np.random.RandomState(seed)
    max_attempts_per_agent = 500
    env = Environment(sim_cfg_local)
    placed = []
    for _ in range(N):
        for attempt in range(max_attempts_per_agent):
            pos = rng.rand(2) * 8 + 1
            if all(np.linalg.norm(pos - p) >= sim_cfg_local.safe_dist for p in placed):
                placed.append(pos)
                vel = (rng.rand(2) - 0.5) * 0.3
                goal = rng.rand(2) * 8 + 1
                env.add_agent(pos, vel, goal)
                break
        else:
            # не удалось разместить агента – меняем seed и пробуем заново всю сцену
            return generate_scene(N, seed + 10000)
    # препятствия
    env.add_obstacle(3.0, 3.0, sim_cfg_local.obs_radius)
    env.add_obstacle(7.0, 7.0, sim_cfg_local.obs_radius)
    # проверка расстояний до препятствий
    for ag in env.agents:
        for obs in env.obstacles:
            if np.linalg.norm(ag.state[:2] - obs.pos) < sim_cfg_local.safe_dist + obs.radius:
                return generate_scene(N, seed + 10000)
    return env, seed

def save_trajectory(env, N, seed, ctrl_name):
    filename = os.path.join(trajectories_dir, f'N{N}_seed{seed}_{ctrl_name}.csv')
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step'] + [f'agent_{i}_x' for i in range(N)] +
                        [f'agent_{i}_y' for i in range(N)] +
                        [f'agent_{i}_vx' for i in range(N)] +
                        [f'agent_{i}_vy' for i in range(N)])
        for step in range(len(env.agents[0].history)):
            row = [step]
            for ag in env.agents:
                state = ag.history[step]
                row.extend([state[0], state[1], state[2], state[3]])
            writer.writerow(row)

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['N', 'seed', 'controller', 'success', 'fraction_reached',
                     'steps', 'collision_pairs_cumulative', 'collision_time',
                     'min_dist', 'avg_acc', 'time_per_step_ms', 'avg_neighbors',
                     'avg_remaining_dist', 'end_speed', 'qp_infeasible_fraction',
                     'saturation_fraction'])

    for N in N_list:
        print(f"\n=== N={N} ===")
        for seed in range(num_seeds):
            base_env, used_seed = generate_scene(N, seed)
            np.random.seed(used_seed)
            random.seed(used_seed)

            save_traj = (seed in representative_seeds.get(N, []))

            for ctrl_name in ['CBF', 'Crystal', 'GCBF+']:
                env = Environment(sim_cfg)
                for ag in base_env.agents:
                    env.add_agent(ag.state[:2].copy(), ag.state[2:4].copy(), ag.goal.copy())
                for obs in base_env.obstacles:
                    env.add_obstacle(obs.pos[0], obs.pos[1], obs.radius)

                if ctrl_name == 'CBF':
                    ctrl = CBFController(cbf_cfg, sim_cfg)
                elif ctrl_name == 'Crystal':
                    ctrl = CrystalController(crystal_cfg, sim_cfg)
                elif ctrl_name == 'GCBF+':
                    ctrl = GCBFController(gnn_cfg, sim_cfg, model_path=GCBF_MODEL_PATH)
                else:
                    continue

                t_per_step = measure_time(ctrl, env)

                collision_time = 0
                total_neighbors = 0.0
                cumulative_coll_pairs = set()
                step_count = 0
                qp_infeasible = 0
                saturation_steps = 0

                for step in range(max_steps):
                    step_count += 1
                    actions = ctrl.compute(env)

                    if ctrl_name == 'CBF':
                        if hasattr(ctrl, 'last_slack') and ctrl.last_slack > 1e-3:
                            qp_infeasible += 1
                        if hasattr(ctrl, 'last_saturated') and any(ctrl.last_saturated):
                            saturation_steps += 1

                    for ag, u in zip(env.agents, actions):
                        ag.update(u, env.config)

                    if hasattr(ctrl, 'last_num_neighbors'):
                        total_neighbors += ctrl.last_num_neighbors

                    if env.collision_pairs() > 0:
                        collision_time += 1
                        for i in range(len(env.agents)):
                            for j in range(i+1, len(env.agents)):
                                d = np.linalg.norm(env.agents[i].state[:2] - env.agents[j].state[:2])
                                if d < env.config.safe_dist:
                                    cumulative_coll_pairs.add(('agent', min(i,j), max(i,j)))
                        for i, ag in enumerate(env.agents):
                            for k, obs in enumerate(env.obstacles):
                                d = np.linalg.norm(ag.state[:2] - obs.pos) - obs.radius
                                if d < env.config.safe_dist:
                                    cumulative_coll_pairs.add(('obs', i, k))

                    if env.all_goals_reached():
                        break

                success = 1 if env.all_goals_reached() else 0
                n_reached = sum(1 for ag in env.agents if np.linalg.norm(ag.state[:2] - ag.goal) < sim_cfg.goal_tolerance)
                fraction_reached = n_reached / len(env.agents) if env.agents else 1.0
                coll_pairs_count = len(cumulative_coll_pairs)
                min_dist = env.min_safety_distance()

                remaining_dists = []
                for ag in env.agents:
                    dist = np.linalg.norm(ag.state[:2] - ag.goal)
                    if dist >= sim_cfg.goal_tolerance:
                        remaining_dists.append(dist)
                avg_remaining_dist = np.mean(remaining_dists) if remaining_dists else 0.0

                end_speeds = []
                for ag in env.agents:
                    if len(ag.history) >= 50:
                        recent_vel = np.array([ag.history[-i][2:4] for i in range(50, 0, -1)])
                        end_speeds.append(np.mean(np.linalg.norm(recent_vel, axis=1)))
                    elif len(ag.history) > 0:
                        end_speeds.append(np.linalg.norm(ag.state[2:4]))
                avg_end_speed = np.mean(end_speeds) if end_speeds else 0.0

                avg_acc = 0.0
                for ag in env.agents:
                    if len(ag.history) > 1:
                        dv = ag.history[-1][2:4] - ag.history[-2][2:4]
                        avg_acc += np.linalg.norm(dv) / env.config.dt
                avg_acc /= len(env.agents) if env.agents else 1

                avg_neighbors = total_neighbors / step_count if step_count > 0 else 0.0
                qp_inf_frac = qp_infeasible / step_count if step_count > 0 else 0.0
                sat_frac = saturation_steps / step_count if step_count > 0 else 0.0

                writer.writerow([N, used_seed, ctrl_name, success, fraction_reached,
                                 step_count, coll_pairs_count, collision_time,
                                 min_dist, avg_acc, t_per_step, avg_neighbors,
                                 avg_remaining_dist, avg_end_speed, qp_inf_frac,
                                 sat_frac])

                print(f"  {ctrl_name} (seed {used_seed}): succ={success}, frac={fraction_reached:.2f}, steps={step_count}, "
                      f"coll_pairs={coll_pairs_count}, coll_time={collision_time}, min_dist={min_dist:.3f}, "
                      f"time={t_per_step:.3f} ms, neigh={avg_neighbors:.1f}, "
                      f"rem_dist={avg_remaining_dist:.3f}, end_spd={avg_end_speed:.3f}, "
                      f"qp_inf_frac={qp_inf_frac:.3f}, sat_frac={sat_frac:.3f}")

                if save_traj:
                    save_trajectory(env, N, used_seed, ctrl_name)

print(f"\nРезультаты сохранены в {output_csv}")
print(f"Траектории представительных сцен сохранены в папке {trajectories_dir}")