"""Генерация датасета и обучение GCBF+ с управлением экспериментами."""
import sys
sys.path.insert(0, '.')
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import random
import time
from src.config import SimConfig, CBFConfig, GNNConfig
from src.environment import Environment
from src.controllers.cbf import CBFController
from src.controllers.gcbf import GCBFController
from src.gcbf_model import GCBFPlusModel


def generate_training_data(num_scenes, save_path, sim_cfg, cbf_cfg, gnn_cfg, max_examples=2_000_000):
    """Генерирует сцены и сохраняет датасет в save_path."""
    if os.path.exists(save_path):
        print(f"Датасет уже существует: {save_path}, генерация пропущена.")
        return

    print(f"Генерация {num_scenes} сцен...")
    cbf_ctrl = CBFController(cbf_cfg, sim_cfg)
    dummy_gcbf = GCBFController.__new__(GCBFController)
    dummy_gcbf.gnn_cfg = gnn_cfg
    dummy_gcbf.sim_cfg = sim_cfg
    dummy_gcbf.node_mean = torch.zeros(4)
    dummy_gcbf.node_std = torch.ones(4)
    dummy_gcbf.edge_mean = torch.zeros(4)
    dummy_gcbf.edge_std = torch.ones(4)

    node_feats_list = []
    edge_indices_list = []
    edge_attrs_list = []
    u_targets_list = []
    scene_ids_list = []
    all_node_feats_all = []
    all_edge_attrs_global = []
    total_examples = 0
    start_time = time.time()

    for scene_idx in range(num_scenes):
        N = np.random.randint(2, 7)      # 2–6 агентов
        M = np.random.randint(0, 5)      # 0–4 препятствий
        env = Environment(sim_cfg)
        for _ in range(N):
            pos = np.random.rand(2) * 8 + 1
            vel = (np.random.rand(2) - 0.5) * 0.8
            goal = np.random.rand(2) * 8 + 1
            env.add_agent(pos, vel, goal)
        for _ in range(M):
            pos = np.random.rand(2) * 8 + 1
            env.add_obstacle(pos[0], pos[1], sim_cfg.obs_radius)

        u_all = cbf_ctrl.compute(env)
        for i, ag in enumerate(env.agents):
            graph_data = dummy_gcbf._build_local_graph(ag, env)
            if graph_data is None:
                continue
            node_feats, edge_index, edge_attr = graph_data
            node_feats_list.append(node_feats.numpy())
            edge_indices_list.append(edge_index.numpy())
            edge_attrs_list.append(edge_attr.numpy())
            u_targets_list.append(u_all[i])
            scene_ids_list.append(scene_idx)
            all_node_feats_all.append(node_feats.numpy())
            all_edge_attrs_global.append(edge_attr.numpy())
            total_examples += 1

        if total_examples >= max_examples:
            print(f"Достигнут лимит примеров ({max_examples}), остановка на сцене {scene_idx+1}")
            break

        if (scene_idx + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"Сгенерировано {scene_idx+1} сцен, примеров: {total_examples}, время: {elapsed:.1f} сек")

    # Статистики нормализации
    print("Вычисление статистик нормализации...")
    all_nodes = np.concatenate(all_node_feats_all, axis=0)
    mean_node = np.mean(all_nodes[:, :4], axis=0)
    std_node = np.std(all_nodes[:, :4], axis=0) + 1e-8
    all_edges = np.concatenate(all_edge_attrs_global, axis=0)
    mean_edge = np.mean(all_edges, axis=0)
    std_edge = np.std(all_edges, axis=0) + 1e-8

    gnn_cfg.norm_mean = mean_node.tolist()
    gnn_cfg.norm_std = std_node.tolist()
    gnn_cfg.edge_mean = mean_edge.tolist()
    gnn_cfg.edge_std = std_edge.tolist()

    # Нормализация
    node_norm_list, edge_norm_list = [], []
    for nf, ea in zip(node_feats_list, edge_attrs_list):
        nf_norm = nf.copy()
        nf_norm[:, :4] = (nf[:, :4] - mean_node) / std_node
        node_norm_list.append(nf_norm)
        edge_norm_list.append((ea - mean_edge) / std_edge)

    # Упаковка в object arrays
    node_arr = np.empty(len(node_norm_list), dtype=object)
    for i, arr in enumerate(node_norm_list):
        node_arr[i] = arr
    edge_idx_arr = np.empty(len(edge_indices_list), dtype=object)
    for i, arr in enumerate(edge_indices_list):
        edge_idx_arr[i] = arr
    edge_attr_arr = np.empty(len(edge_norm_list), dtype=object)
    for i, arr in enumerate(edge_norm_list):
        edge_attr_arr[i] = arr
    u_arr = np.array(u_targets_list, dtype=np.float32)

    np.savez(save_path,
             node_feats=node_arr,
             edge_index=edge_idx_arr,
             edge_attr=edge_attr_arr,
             u_targets=u_arr,
             scene_ids=np.array(scene_ids_list, dtype=np.int64),
             norm_mean=mean_node, norm_std=std_node,
             edge_mean=mean_edge, edge_std=std_edge)
    print(f"Датасет сохранён: {save_path}, примеров: {total_examples}")


def train_gcbf(data_path, model_save_path, gnn_cfg, device='cpu'):
    """Обучение модели с train/val split и early stopping."""
    data = np.load(data_path, allow_pickle=True)
    node_feats_arr = data['node_feats']
    edge_indices_arr = data['edge_index']
    edge_attrs_arr = data['edge_attr']
    u_targets = torch.tensor(data['u_targets'], dtype=torch.float32)
    scene_ids = data['scene_ids']
    if 'edge_mean' in data:
        gnn_cfg.edge_mean = data['edge_mean'].tolist()
        gnn_cfg.edge_std = data['edge_std'].tolist()
        gnn_cfg.norm_mean = data['norm_mean'].tolist()
        gnn_cfg.norm_std = data['norm_std'].tolist()

    # Train/Val split по сценам
    unique_scenes = np.unique(scene_ids)
    rng = np.random.RandomState(42)
    rng.shuffle(unique_scenes)
    n_val = int(0.15 * len(unique_scenes))
    val_scenes = set(unique_scenes[:n_val])
    train_idx = [i for i, s in enumerate(scene_ids) if s not in val_scenes]
    val_idx   = [i for i, s in enumerate(scene_ids) if s in val_scenes]
    print(f"Train примеров: {len(train_idx)}, Val примеров: {len(val_idx)}")

    model = GCBFPlusModel(node_feat_dim=8, edge_feat_dim=4, hidden=128, dropout=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    mse_loss = torch.nn.MSELoss()
    best_val_mae = float('inf')
    epochs_no_improve = 0
    patience_es = 30

    for epoch in range(gnn_cfg.num_epochs):
        # Training
        model.train()
        random.shuffle(train_idx)
        train_loss = 0.0
        train_mae = 0.0
        n_train = len(train_idx)
        for start in range(0, n_train, gnn_cfg.batch_size):
            batch_idx = train_idx[start:start+gnn_cfg.batch_size]
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_mae = 0.0
            for idx in batch_idx:
                node_feat = torch.tensor(node_feats_arr[idx], dtype=torch.float32).to(device)
                edge_index = torch.tensor(edge_indices_arr[idx], dtype=torch.long).to(device)
                edge_attr = torch.tensor(edge_attrs_arr[idx], dtype=torch.float32).to(device)
                u_target = u_targets[idx].to(device)
                u_pred = model(node_feat, edge_index, edge_attr)
                loss = mse_loss(u_pred[0], u_target)
                batch_loss += loss
                batch_mae += torch.abs(u_pred[0] - u_target).mean().item()
            (batch_loss / len(batch_idx)).backward()
            optimizer.step()
            train_loss += batch_loss.item()
            train_mae += batch_mae
        avg_train_loss = train_loss / n_train
        avg_train_mae = train_mae / n_train

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for idx in val_idx:
                node_feat = torch.tensor(node_feats_arr[idx], dtype=torch.float32).to(device)
                edge_index = torch.tensor(edge_indices_arr[idx], dtype=torch.long).to(device)
                edge_attr = torch.tensor(edge_attrs_arr[idx], dtype=torch.float32).to(device)
                u_target = u_targets[idx].to(device)
                u_pred = model(node_feat, edge_index, edge_attr)
                loss = mse_loss(u_pred[0], u_target)
                val_loss += loss.item()
                val_mae += torch.abs(u_pred[0] - u_target).mean().item()
        avg_val_loss = val_loss / len(val_idx)
        avg_val_mae = val_mae / len(val_idx)

        scheduler.step(avg_val_loss)
        print(f"Эпоха {epoch+1}/{gnn_cfg.num_epochs} | Train Loss: {avg_train_loss:.4f} MAE: {avg_train_mae:.4f} | Val Loss: {avg_val_loss:.4f} MAE: {avg_val_mae:.4f}")

        # Early stopping + checkpoint
        if avg_val_mae < best_val_mae - 1e-4:
            best_val_mae = avg_val_mae
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  Сохранена лучшая модель (Val MAE={best_val_mae:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_es:
                print(f"Early stopping на эпохе {epoch+1}")
                break

    print(f"Обучение завершено. Лучшая Val MAE: {best_val_mae:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Обучение GCBF+")
    parser.add_argument('--exp_name', type=str, required=True, help="Название эксперимента (папка)")
    parser.add_argument('--num_scenes', type=int, default=50000, help="Количество сцен")
    parser.add_argument('--max_examples', type=int, default=2_000_000, help="Макс. число обучающих примеров")
    parser.add_argument('--num_epochs', type=int, default=300, help="Число эпох")
    parser.add_argument('--batch_size', type=int, default=16, help="Размер батча")
    args = parser.parse_args()

    # Создаём папку эксперимента
    exp_dir = os.path.join('experiments', args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    sim_cfg = SimConfig()
    cbf_cfg = CBFConfig()
    gnn_cfg = GNNConfig(num_train_scenes=args.num_scenes,
                        num_epochs=args.num_epochs,
                        batch_size=args.batch_size,
                        hidden_dim=128)

    data_path = os.path.join(exp_dir, 'training_data.npz')
    model_path = os.path.join(exp_dir, 'gcbf_model.pt')

    generate_training_data(
        num_scenes=args.num_scenes,
        save_path=data_path,
        sim_cfg=sim_cfg,
        cbf_cfg=cbf_cfg,
        gnn_cfg=gnn_cfg,
        max_examples=args.max_examples
    )

    train_gcbf(
        data_path=data_path,
        model_save_path=model_path,
        gnn_cfg=gnn_cfg,
        device='cpu'
    )