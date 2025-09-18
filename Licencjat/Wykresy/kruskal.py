#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from itertools import combinations

# CONFIGURATION
base_folder = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/"

directories = [
    os.path.join(base_folder, "data", "test_output", "cohorts", "c1", "alt"),
    os.path.join(base_folder, "data", "test_output", "cohorts", "c2", "alt"),
    os.path.join(base_folder, "data", "test_output", "cohorts", "c3", "alt"),
    os.path.join(base_folder, "data", "test_output", "cohorts", "c4", "alt"),
    os.path.join(base_folder, "data", "test_output", "cohorts", "c5", "alt"),
    os.path.join(base_folder, "data", "test_output", "cohorts", "c6", "alt"),
    os.path.join(base_folder, "data", "test_output", "cohorts", "c7", "alt"),
    os.path.join(base_folder, "data", "test_output", "cohorts", "c8", "alt"),
    os.path.join(base_folder, "data", "test_output", "cohorts", "c9", "alt"),
    os.path.join(base_folder, "data", "test_output", "cohorts", "c10", "alt"),
]

prefixes = ["020", "040", "060", "080"]
percent_labels = {"020": "20", "040": "40", "060": "60", "080": "80"}
n_models_per_prefix = 5

classes = ["PA", "NA", "PI", "NI"]

out_png = "auc_boxplots_by_class_colored.png"
out_report = "auc_kruskal_and_pairwise_report.txt"

# Colours for boxplots
percent_colors = {
    "20": "#19a2b8",  # green
    "40": "#1ba04e",  # orange
    "60": "#b5ca6f",  # purple
    "80": "#e7c429",  # pink
}


def safe_parse_list(s):
    """
    Sparsowanie stringu reprezentujący listę (np. '[0.9,0.8,...]') -> list(float)
    """
    try:
        if isinstance(s, str):
            s_clean = s.replace('np.float64(', '').replace(')', '')
            parsed = ast.literal_eval(s_clean)
            return [float(x) for x in parsed]
        elif isinstance(s, (list, tuple, np.ndarray)):
            return [float(x) for x in s]
    except Exception:
        return None
    return None

def read_best_mcc_from_dir(model_dir):
    try:
        df = pd.read_csv(model_dir)
        if "AUC" in df.columns:
            return safe_parse_list(df["AUC"].iloc[0])
    except Exception:
        pass
    return None

# LOAD DATA
grouped = {percent_labels[p]: [[] for _ in range(4)] for p in prefixes}

for prefix in prefixes:
    for model_idx in range(1, n_models_per_prefix + 1):
        for cohort_dir in directories:
            model_dirname = f"high_{prefix}_{model_idx}"
            model_dirpath = os.path.join(cohort_dir, model_dirname, 'metrics', f'best_{model_dirname}_metrics.csv')
            auc_vals = read_best_mcc_from_dir(model_dirpath)
            print(model_dirpath)
            print(auc_vals)
            label = percent_labels[prefix]
            if auc_vals is None:
                for ci in range(4):
                    grouped[label][ci].append(np.nan)
            else:
                for ci in range(4):
                    val = auc_vals[ci] if ci < len(auc_vals) else np.nan
                    grouped[label][ci].append(float(val))

# Delete NaNs from each group
for lab in list(grouped.keys()):
    for ci in range(4):
        arr = np.array(grouped[lab][ci], dtype=float)
        arr = arr[~np.isnan(arr)]
        grouped[lab][ci] = arr

# BOXPLOTS
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=True)

for ci, cls in enumerate(classes):
    row, col = divmod(ci, 2)
    ax = axes[row, col]
    data_list = [grouped[percent_labels[p]][ci] for p in prefixes]
    positions = np.arange(1, len(prefixes) + 1)
    widths = 0.6
    box_facecolors = [percent_colors[percent_labels[p]] for p in prefixes]
    bp = ax.boxplot(data_list, positions=positions, widths=widths, patch_artist=True,
                    showmeans=True, meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white'))

    # Style boxplots
    for patch, color in zip(bp['boxes'], box_facecolors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.2)
    # Style whiskers, caps
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.0)
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.0)
    for flier in bp['fliers']:
        flier.set(marker='x', color='gray', alpha=0.9)

    #  Add grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Add text annotations for n and median
    ymin, ymax = ax.get_ylim() if ax.get_ylim()[0] != ax.get_ylim()[1] else (0,1)
    y_text = ymin + 0.05 * (ymax - ymin)
    for i, arr in enumerate(data_list):
        pos = positions[i]
        n = len(arr)
        med = np.median(arr) if n > 0 else np.nan
        txt = f"n={n}\nmed={med:.3f}" if n > 0 else "n=0\nmed=NA"

    ax.set_title(f"{cls}")
    ax.set_xticks(positions)
    ax.set_xticklabels([percent_labels[p] for p in prefixes])
    ax.set_xlabel("Procent reaktywowanych filtrów", fontsize=18)
    if ci == 0:
        ax.set_ylabel("AUC", fontsize=18)
    if ci == 1:
        ax.set_ylabel("AUC", fontsize=18)

# Legend
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=percent_colors[p], edgecolor='black', label=f"{p}%") for p in ["20","40","60","80"]]
fig.legend(handles=legend_patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02), frameon=True)

fig.suptitle("Rozkład AUC na zbiorze Alt modeli z reaktywowanymi filtrami", fontsize=20, weight='bold')
plt.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Zapisano wykres: {out_png}")

# STATISTICAL TESTS
with open(out_report, "w", encoding="utf-8") as fh:
    fh.write("Kruskal-Wallis per class and pairwise Mann-Whitney U (BH corrected if applied)\n")
    fh.write("Groups: 20,40,60,80 (prefixes 020,040,060,080)\n\n")

    for ci, cls in enumerate(classes):
        fh.write(f"CLASS: {cls}\n")
        group_arrays = [grouped[percent_labels[p]][ci] for p in prefixes]
        for lab, arr in zip([percent_labels[p] for p in prefixes], group_arrays):
            fh.write(f"Group {lab}: n={len(arr)}, median={np.median(arr) if len(arr)>0 else 'NA'}\n")
        fh.write("\n")

        # Kruskal-Wallis
        nonempty_counts = sum(1 for arr in group_arrays if len(arr) > 0)
        if nonempty_counts < 2:
            fh.write("Brak wystarczających danych do testu Kruskala-Wallisa (mniej niż 2 niepuste grupy).\n\n")
            continue

        try:
            arrays_for_kw = [arr for arr in group_arrays if len(arr) > 0]
            stat, p_kw = kruskal(*arrays_for_kw)
            fh.write(f"Kruskal-Wallis: H={stat:.4f}, p={p_kw:.6e}\n")
        except Exception as e:
            fh.write(f"Kruskal-Wallis: błąd wykonania testu: {e}\n")
            fh.write("\n")
            continue

        pairs = list(combinations(range(len(prefixes)), 2))
        raw_pvals = []
        pair_info = []
        for i, j in pairs:
            arr1 = group_arrays[i]
            arr2 = group_arrays[j]
            if len(arr1) < 2 or len(arr2) < 2:
                raw_pvals.append(np.nan)
                pair_info.append((i, j, np.nan, np.nan, "insufficient data"))
                continue
            try:
                stat_pw, p_pw = mannwhitneyu(arr1, arr2, alternative='two-sided')
            except Exception as e:
                stat_pw, p_pw = np.nan, np.nan
            raw_pvals.append(p_pw)
            pair_info.append((i, j, stat_pw, p_pw, "ok"))

        # korekta BH
        pvals_for_corr = [pv if not np.isnan(pv) else 1.0 for pv in raw_pvals]
        rej, pvals_bh, _, _ = multipletests(pvals_for_corr, alpha=0.05, method='fdr_bh')

        fh.write("Pair\tstat\tp_raw\tp_BH\treject_BH\n")
        for idx_pair, (i, j, stat_pw, p_pw, status) in enumerate(pair_info):
            pair_label = f"{percent_labels[prefixes[i]]} vs {percent_labels[prefixes[j]]}"
            if np.isnan(p_pw):
                fh.write(f"{pair_label}\t-\tNA\tNA\tinsufficient data\n")
            else:
                fh.write(f"{pair_label}\t{stat_pw:.4f}\t{p_pw:.6e}\t{pvals_bh[idx_pair]:.6e}\t{bool(rej[idx_pair])}\n")

        # KW result summary
        fh.write(f"\nKruskal-Wallis result: p = {p_kw:.6e} -> {'istotny (p<0.05)' if p_kw < 0.05 else 'nieistotny (p>=0.05)'}\n")
        if p_kw >= 0.05:
            fh.write("Uwaga: Kruskal-Wallis nie wykazał istotnej różnicy między grupami; wyniki parowych testów należy interpretować ostrożnie.\n")
        fh.write("\n" + ("-"*80) + "\n\n")

print(f"Zapisano raport: {out_report}")
