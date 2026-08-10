"""Контроллер GCBF+ с локальным графом."""
import numpy as np
import torch
from src.config import GNNConfig, SimConfig, NodeType
from src.environment import Environment
from src.controllers.base import BaseController
from src.gcbf_model import GCBFPlusModel

class GCBFController(BaseController):
    def __init__(self, gnn_cfg: GNNConfig, sim_cfg: SimConfig, model_path='gcbf_model.pt', device='cpu'):
        self.gnn_cfg = gnn_cfg
        self.sim_cfg = sim_cfg
        self.device = torch.device(device)
        self.model = GCBFPlusModel(node_feat_dim=8, edge_feat_dim=4, hidden=gnn_cfg.hidden_dim)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.node_mean = torch.tensor(gnn_cfg.norm_mean if gnn_cfg.norm_mean else [0.]*4, dtype=torch.float32)
        self.node_std = torch.tensor(gnn_cfg.norm_std if gnn_cfg.norm_std else [1.]*4, dtype=torch.float32)
        self.edge_mean = torch.tensor(gnn_cfg.edge_mean if gnn_cfg.edge_mean else [0.]*4, dtype=torch.float32)
        self.edge_std = torch.tensor(gnn_cfg.edge_std if gnn_cfg.edge_std else [1.]*4, dtype=torch.float32)
        self.last_num_neighbors = 0.0

    def compute(self, env: Environment) -> list:
        actions = []
        total_neighbors = 0
        for ag in env.agents:
            graph_data = self._build_local_graph(ag, env)
            if graph_data is None:
                actions.append(np.zeros(2))
                continue
            node_feats, edge_index, edge_attr = graph_data
            num_neighbors = node_feats.shape[0] - 2  # исключаем себя и цель
            total_neighbors += num_neighbors
            node_feats = node_feats.to(self.device)
            edge_index = edge_index.to(self.device)
            edge_attr = edge_attr.to(self.device)
            with torch.no_grad():
                u_pred = self.model(node_feats, edge_index, edge_attr)
            u = u_pred[0].cpu().numpy()
            u = np.clip(u, -self.sim_cfg.u_max, self.sim_cfg.u_max)
            actions.append(u)
        avg_neighbors = total_neighbors / len(env.agents) if env.agents else 0
        self.last_num_neighbors = avg_neighbors
        return actions

    def _build_local_graph(self, agent, env: Environment):
        pos = agent.state[:2]
        vel = agent.state[2:4]
        nodes = []
        onehot_self = [1.0, 0.0, 0.0, 0.0]
        nodes.append(np.array([pos[0], pos[1], vel[0], vel[1]] + onehot_self))
        goal = agent.goal
        onehot_goal = [0.0, 0.0, 1.0, 0.0]
        nodes.append(np.array([goal[0], goal[1], 0.0, 0.0] + onehot_goal))
        edges_src, edges_dst, edge_attrs = [], [], []
        edges_src.append(1)
        edges_dst.append(0)
        edge_attrs.append(np.concatenate([goal - pos, vel]))

        onehot_agent = [0.0, 1.0, 0.0, 0.0]
        for other in env.agents:
            if other.id == agent.id: continue
            d = np.linalg.norm(other.state[:2] - pos)
            if d < self.gnn_cfg.r_sensor:
                node_idx = len(nodes)
                nodes.append(np.array([other.state[0], other.state[1], other.state[2], other.state[3]] + onehot_agent))
                edges_src.append(node_idx)
                edges_dst.append(0)
                rel_pos = other.state[:2] - pos
                rel_vel = other.state[2:4] - vel
                edge_attrs.append(np.concatenate([rel_pos, rel_vel]))

        onehot_obs = [0.0, 0.0, 0.0, 1.0]
        for obs in env.obstacles:
            d = np.linalg.norm(obs.pos - pos)
            if d < self.gnn_cfg.r_sensor:
                node_idx = len(nodes)
                nodes.append(np.array([obs.pos[0], obs.pos[1], 0.0, 0.0] + onehot_obs))
                edges_src.append(node_idx)
                edges_dst.append(0)
                rel_pos = obs.pos - pos
                edge_attrs.append(np.concatenate([rel_pos, np.zeros(2)]))

        if len(edges_src) == 0:
            return None

        node_feats = torch.tensor(np.array(nodes), dtype=torch.float32)
        node_feats[:, :4] = (node_feats[:, :4] - self.node_mean) / self.node_std
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float32)
        edge_attr = (edge_attr - self.edge_mean) / self.edge_std
        return node_feats, edge_index, edge_attr