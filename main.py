"""Точка входа: генерация данных, обучение GCBF+, запуск экспериментов, вывод результатов."""
import os
import sys
from src.config import SimConfig, CBFConfig, CrystalConfig, GNNConfig
from src.train_gcbf import generate_training_data, train_gcbf
from src.experiments import run_experiments, print_summary
from src.visualize import plot_boxplots

def main():
    print("=== Шаг 1: Конфигурация ===")
    sim_cfg = SimConfig()
    cbf_cfg = CBFConfig()
    gnn_cfg = GNNConfig(num_train_scenes=500, num_epochs=10)  # для быстроты

    # Генерация данных и обучение, если модель ещё не обучена
    model_path = 'gcbf_model.pt'
    data_path = 'training_data.npz'
    if not os.path.exists(model_path):
        print("Модель не найдена. Начинаем генерацию данных и обучение...")
        generate_training_data(
            num_scenes=gnn_cfg.num_train_scenes,
            sim_cfg=sim_cfg,
            cbf_cfg=cbf_cfg,
            gnn_cfg=gnn_cfg,
            save_path=data_path
        )
        train_gcbf(
            data_path=data_path,
            model_save_path=model_path,
            gnn_cfg=gnn_cfg,
            device='cpu'
        )
    else:
        print(f"Найдена обученная модель: {model_path}")

    print("\n=== Шаг 2: Запуск экспериментов ===")
    results = run_experiments(seeds=5, agent_counts=[3, 5, 10], use_gcbf=True)
    print_summary(results)

    print("\n=== Шаг 3: Визуализация ===")
    plot_boxplots(results, save_path='comparison_boxplot.png')
    print("Готово! Результаты сохранены.")

if __name__ == '__main__':
    main()