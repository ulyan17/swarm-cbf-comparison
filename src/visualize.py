"""Визуализация результатов экспериментов."""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

def plot_boxplots(results: Dict[str, List[Dict]], save_path: str = 'comparison_boxplot.png'):
    """Строит ящиковые диаграммы для основных метрик."""
    controller_names = list(results.keys())
    metrics = ['steps', 'collisions', 'min_dist', 'avg_acc']
    metric_labels = ['Шагов до цели', 'Столкновений', 'Мин. расстояние', 'Среднее ускорение']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, metric, label in zip(axes, metrics, metric_labels):
        data = []
        for name in controller_names:
            values = [m[metric] for m in results[name]]
            data.append(values)
        ax.boxplot(data, labels=controller_names)
        ax.set_title(label)
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_trajectories(env_history, save_path: str = 'trajectories.png'):
    """Рисует траектории агентов (для одного эпизода). env_history - список Environment."""
    # Заглушка
    pass