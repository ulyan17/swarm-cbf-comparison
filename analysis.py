"""Расширенный анализ результатов сравнения CBF, Crystal, GCBF+."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, friedmanchisquare
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar

# ------------------------------------------------------------
# Загрузка данных
# ------------------------------------------------------------
df = pd.read_csv('comparison_results.csv')
controllers = ['CBF', 'Crystal', 'GCBF+']
# Обновлённые метрики (включая новые)
metrics = ['collision_pairs_cumulative', 'collision_time', 'min_dist',
           'avg_acc', 'time_per_step_ms', 'fraction_reached',
           'avg_remaining_dist', 'end_speed']
N_values = [5, 10, 20]

# ------------------------------------------------------------
# Функция для парных значений (явный merge по seed)
# ------------------------------------------------------------
def paired_values(sub, c1, c2, metric):
    """Возвращает два aligned массива для парного сравнения."""
    d1 = sub[sub['controller'] == c1][['seed', metric]].rename(columns={metric: 'v1'})
    d2 = sub[sub['controller'] == c2][['seed', metric]].rename(columns={metric: 'v2'})
    merged = d1.merge(d2, on='seed', how='inner')
    return merged['v1'].values, merged['v2'].values

# ------------------------------------------------------------
# 1. Описательная статистика (таблица для статьи)
# ------------------------------------------------------------
print("Средние значения по N=20:")
summary_20 = df[df['N'] == 20].groupby('controller').agg(
    success_mean=('success', 'mean'),
    fraction_mean=('fraction_reached', 'mean'),
    steps_mean=('steps', 'mean'),
    coll_pairs_mean=('collision_pairs_cumulative', 'mean'),
    coll_time_mean=('collision_time', 'mean'),
    min_dist_mean=('min_dist', 'mean'),
    accel_mean=('avg_acc', 'mean'),
    time_mean=('time_per_step_ms', 'mean'),
    remaining_dist_mean=('avg_remaining_dist', 'mean'),
    end_speed_mean=('end_speed', 'mean')
).round(3)
print(summary_20)
summary_20.to_csv('summary_table_N20.csv')
summary_20.to_latex('summary_table_N20.tex')
print("Таблица сохранена в summary_table_N20.csv и .tex")

# ------------------------------------------------------------
# 2. Статистические тесты (Friedman + Wilcoxon с глобальной FDR)
# ------------------------------------------------------------
print("\n=== Статистические тесты ===")
all_pvalues = []  # для глобальной FDR-коррекции

for N in N_values:
    sub = df[df['N'] == N]
    print(f"\n--- N={N} ---")
    for metric in metrics:
        # Friedman test
        groups = [sub[sub['controller'] == c][metric].values for c in controllers]
        try:
            stat, p = friedmanchisquare(*groups)
            print(f"Friedman test for {metric}: stat={stat:.3f}, p={p:.4f}")
        except Exception as e:
            print(f"Friedman test for {metric} failed: {e}")
            p = 1.0

        if p < 0.05:
            print("  Post-hoc Wilcoxon (paired, aligned by seed):")
            for i, c1 in enumerate(controllers):
                for c2 in controllers[i+1:]:
                    try:
                        v1, v2 = paired_values(sub, c1, c2, metric)
                        if len(v1) == 0 or np.allclose(v1, v2):
                            print(f"    {c1} vs {c2}: all zero differences, skipped")
                            continue
                        wp = wilcoxon(v1, v2)
                        all_pvalues.append((N, metric, c1, c2, wp.pvalue))
                        print(f"    {c1} vs {c2}: p={wp.pvalue:.4f}")
                    except Exception as e:
                        print(f"    {c1} vs {c2}: error ({e})")

# Глобальная FDR-коррекция (Benjamini-Hochberg)
if all_pvalues:
    raw_pvals = [x[4] for x in all_pvalues]
    _, pvals_corrected, _, _ = multipletests(raw_pvals, alpha=0.05, method='fdr_bh')
    print("\n=== Глобальная FDR-коррекция (Benjamini-Hochberg) ===")
    for (N, metric, c1, c2, raw_p), corr_p in zip(all_pvalues, pvals_corrected):
        sig = "*" if corr_p < 0.05 else ""
        print(f"  N={N}, {metric}, {c1} vs {c2}: raw p={raw_p:.4f}, corr p={corr_p:.4f} {sig}")

# ------------------------------------------------------------
# 3. McNemar test для success (все N)
# ------------------------------------------------------------
print("\n=== McNemar test for success (paired by seed) ===")
mcnemar_rows = []
for N in N_values:
    sub = df[df['N'] == N]
    print(f" N={N}:")
    for c1, c2 in [('CBF','Crystal'), ('CBF','GCBF+'), ('Crystal','GCBF+')]:
        s1 = sub[sub['controller'] == c1][['seed', 'success']].rename(columns={'success': 's1'})
        s2 = sub[sub['controller'] == c2][['seed', 'success']].rename(columns={'success': 's2'})
        merged = s1.merge(s2, on='seed', how='inner')
        a = ((merged['s1'] == 1) & (merged['s2'] == 0)).sum()
        b = ((merged['s1'] == 0) & (merged['s2'] == 1)).sum()
        if a + b > 0:
            table = [[0, a], [b, 0]]
            result = mcnemar(table, exact=False, correction=True)
            print(f"    {c1} vs {c2}: disc_a={a}, disc_b={b}, stat={result.statistic:.3f}, p={result.pvalue:.4f}")
            mcnemar_rows.append([N, c1, c2, a, b, result.statistic, result.pvalue])
        else:
            print(f"    {c1} vs {c2}: no discordant pairs")
            mcnemar_rows.append([N, c1, c2, 0, 0, np.nan, 1.0])
pd.DataFrame(mcnemar_rows, columns=['N','method1','method2','disc_a','disc_b','statistic','pvalue']).to_csv('mcnemar_results.csv', index=False)

# ------------------------------------------------------------
# 4. CBF-специфичные метрики (feasibility, saturation)
# ------------------------------------------------------------
print("\n=== CBF feasibility & saturation analysis ===")
cbf_data = df[df['controller'] == 'CBF'].copy()
cbf_summary = cbf_data.groupby('N')[['qp_infeasible_fraction', 'saturation_fraction']].agg(['mean','std','max'])
cbf_summary.to_csv('cbf_feasibility_summary.csv')
print(cbf_summary)

# Корреляция с безопасностью
for metric in ['qp_infeasible_fraction', 'saturation_fraction']:
    corr_min = cbf_data[metric].corr(cbf_data['min_dist'])
    corr_coll = cbf_data[metric].corr(cbf_data['collision_pairs_cumulative'])
    print(f"  {metric}: corr with min_dist={corr_min:.3f}, corr with collision_pairs={corr_coll:.3f}")

# ------------------------------------------------------------
# 5. Анализ GCBF+ по числу соседей
# ------------------------------------------------------------
print("\n=== GCBF+ neighbors vs success ===")
gnn_data = df[df['controller'] == 'GCBF+'].copy()
gnn_data['neighbor_bin'] = pd.cut(gnn_data['avg_neighbors'],
                                  bins=[0, 4, 6, 8, 20],
                                  labels=['<4', '4-6', '6-8', '>8'])
neighbor_analysis = gnn_data.groupby('neighbor_bin', observed=False).agg(
    success_rate=('success', 'mean'),
    fraction_mean=('fraction_reached', 'mean'),
    collision_pairs=('collision_pairs_cumulative', 'mean'),
    count=('success', 'count')
).round(3)
print(neighbor_analysis)
neighbor_analysis.to_csv('gcbf_neighbors_vs_success.csv')

# ------------------------------------------------------------
# 6. Классификация типов отказов
# ------------------------------------------------------------
def classify_failure(row):
    if row['success'] == 1:
        return 'success'
    if row['end_speed'] < 0.05 and row['fraction_reached'] < 0.5:
        return 'deadlock'
    if row['collision_time'] > row['steps'] / 2:
        return 'collision'
    return 'timeout_other'

df['failure_mode'] = df.apply(classify_failure, axis=1)
failure_breakdown = df[df['success'] == 0].groupby(['N', 'controller', 'failure_mode']).size().unstack(fill_value=0)
failure_breakdown.to_csv('failure_mode_breakdown.csv')
print("\n=== Failure mode breakdown ===")
print(failure_breakdown)

# ------------------------------------------------------------
# 7. Среднее число шагов до успеха (только успешные)
# ------------------------------------------------------------
print("\n=== Среднее число шагов до успеха (только успешные) ===")
paired_steps = []
for N in N_values:
    sub = df[df['N'] == N]
    print(f" N={N}:")
    for c1, c2 in [('CBF','Crystal'), ('CBF','GCBF+'), ('Crystal','GCBF+')]:
        # Берём только сцены, где оба метода успешны
        succ1 = sub[(sub['controller']==c1) & (sub['success']==1)][['seed','steps']].rename(columns={'steps':'s1'})
        succ2 = sub[(sub['controller']==c2) & (sub['success']==1)][['seed','steps']].rename(columns={'steps':'s2'})
        merged = succ1.merge(succ2, on='seed', how='inner')
        if len(merged) > 0:
            mean1, mean2 = merged['s1'].mean(), merged['s2'].mean()
            try:
                _, wp = wilcoxon(merged['s1'], merged['s2'])
            except:
                wp = None
            print(f"    {c1} vs {c2}: mean steps {mean1:.1f} vs {mean2:.1f}, paired t-test p={wp if wp else 'N/A'}")
            paired_steps.append([N, c1, c2, len(merged), mean1, mean2, wp])
pd.DataFrame(paired_steps, columns=['N','method1','method2','n_pairs','mean1','mean2','wilcoxon_p']).to_csv('paired_steps_on_success.csv', index=False)

# ------------------------------------------------------------
# 8. Графики
# ------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# 8.1 Boxplots по всем метрикам для каждого N
fig, axes = plt.subplots(len(metrics), len(N_values), figsize=(18, 20))
for i, metric in enumerate(metrics):
    for j, N in enumerate(N_values):
        ax = axes[i, j]
        data = df[df['N'] == N]
        sns.boxplot(x='controller', y=metric, data=data, ax=ax, order=controllers)
        ax.set_title(f'{metric} (N={N})')
        ax.set_xlabel('')
plt.tight_layout()
plt.savefig('boxplots_all_metrics.png', dpi=150)
plt.close()

# 8.2 Время vs N
plt.figure(figsize=(8,5))
for ctrl in controllers:
    means = df[df['controller'] == ctrl].groupby('N')['time_per_step_ms'].mean()
    stds  = df[df['controller'] == ctrl].groupby('N')['time_per_step_ms'].std()
    plt.errorbar(N_values, means, yerr=stds, label=ctrl, capsize=5)
plt.xlabel('Number of agents')
plt.ylabel('Time per step (ms)')
plt.legend()
plt.grid(True)
plt.savefig('time_vs_N.png', dpi=150)
plt.close()

# 8.3 Success rate vs N
plt.figure(figsize=(8,5))
for ctrl in controllers:
    success_rates = df[df['controller'] == ctrl].groupby('N')['success'].mean()
    plt.plot(N_values, [success_rates[N] for N in N_values], 'o-', label=ctrl)
plt.xlabel('Number of agents')
plt.ylabel('Success rate')
plt.legend()
plt.grid(True)
plt.savefig('success_vs_N.png', dpi=150)
plt.close()

# 8.4 GCBF+ число соседей vs успех
if 'avg_neighbors' in df.columns:
    fig, ax1 = plt.subplots(figsize=(8,5))
    gnn = df[df['controller'] == 'GCBF+']
    means = gnn.groupby('N')['avg_neighbors'].mean()
    ax1.plot(N_values, [means[N] for N in N_values], 's-', color='green', label='Avg neighbors')
    ax1.set_xlabel('Number of agents')
    ax1.set_ylabel('Average neighbors', color='green')
    ax1.tick_params(axis='y', labelcolor='green')

    ax2 = ax1.twinx()
    success = gnn.groupby('N')['success'].mean()
    ax2.plot(N_values, [success[N] for N in N_values], 'o-', color='red', label='Success rate (GCBF+)')
    ax2.set_ylabel('Success rate', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.title('GCBF+: neighbors vs success')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('neighbors_vs_success.png', dpi=150)
    plt.close()

# 8.5 CBF-специфичные метрики
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, metric in zip(axes, ['qp_infeasible_fraction', 'saturation_fraction']):
    cbf_df = df[df['controller'] == 'CBF']
    sns.boxplot(x='N', y=metric, data=cbf_df, ax=ax)
    ax.set_title(f'CBF: {metric}')
    ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('cbf_specific_metrics.png', dpi=150)
plt.close()

# 8.6 Диаграмма типов отказов (stacked bar)
failure_counts = df[df['success'] == 0].groupby(['N', 'controller', 'failure_mode']).size().unstack(fill_value=0)
failure_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Failure mode breakdown by method and N')
plt.ylabel('Number of episodes')
plt.legend(title='Failure mode')
plt.tight_layout()
plt.savefig('failure_mode_stacked.png', dpi=150)
plt.close()

print("\nВсе графики и таблицы сохранены.")