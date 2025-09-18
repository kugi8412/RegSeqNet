#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONFIGURATION
base_folder = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/"

metrics_high = os.path.join(base_folder, "test_set", "best_high", "prediction", "highsignal", "metrics")
metrics_alt = os.path.join(base_folder, "test_set", "best_alt", "prediction", "highsignal", "metrics")

directories = [
    os.path.join(base_folder, "data", "test_output", "cohorts", "c8", "alt"),
    os.path.join(base_folder, "data", "test_output", "cohorts", "c9", "alt"),
    os.path.join(base_folder, "data", "test_output", "cohorts", "c10", "alt"),
]

alternative_directories = [
    os.path.join(base_folder, "data", "test_output", "cohorts", "c8", "high"),
    os.path.join(base_folder, "data", "test_output", "cohorts", "c9", "high"),
    os.path.join(base_folder, "data", "test_output", "cohorts", "c10", "high"),
]

prefixes = ["020", "040", "060", "080"]
n_models_per_prefix = 5
n_cohorts = len(directories)

# classes and colours
classes = ["PA", "NA", "PI", "NI"]
class_colors = ["C0", "C1", "C2", "C3"]

comparison_files_a = [
    ("alt1_last_metrics.csv", "alt_last"),
    ("best000_highsignal_metrics.csv", "alt_high_0"),
]

comparison_files_b = [
    ("alt1_last_metrics.csv", "alt_last"),
    ("best000_highsignal_metrics.csv", "alt_high_0"),
]

def read_best_auc_from_dir(model_dir):
    """
    Odczytuje pliki best*.csv w katalogu <model_dir>/metrics i zwraca listę AUC (float).
    Jeśli brak pliku, zwraca None.
    """
    metrics_dir = os.path.join(model_dir, "metrics")
    if not os.path.isdir(metrics_dir):
        return None
    for fname in sorted(os.listdir(metrics_dir)):
        if fname.endswith(".csv") and fname.startswith("best"):
            fpath = os.path.join(metrics_dir, fname)
            try:
                df = pd.read_csv(fpath)
                val = df["AUC"].iloc[0]
                auc_list = eval(val) if isinstance(val, str) else list(val)
                return [float(x) for x in auc_list]
            except Exception:
                continue
    return None

def gather_metrics_for_directories(directories_list):
    """
    Gathers metrics["AUC"] -> 4 lists (one for each class) and x_labels in the desired order:
    for prefix in prefixes:
        for model_idx in 1..n_models_per_prefix:
            for cohort_idx in 0..n_cohorts-1:
                odczyt z <cohort_dir>/high_{prefix}_{model_idx}/metrics/best*.csv
    """
    metrics = {"AUC": [[] for _ in range(4)]}
    x_labels = []
    for prefix in prefixes:
        for model_idx in range(1, n_models_per_prefix + 1):
            for cohort_idx, cohort_dir in enumerate(directories_list):
                model_dirname = f"high_{prefix}_{model_idx}"
                model_dirpath = os.path.join(cohort_dir, model_dirname)
                auc_vals = read_best_auc_from_dir(model_dirpath)
                if auc_vals is None:
                    for ci in range(4):
                        metrics["AUC"][ci].append(np.nan)
                else:
                    for ci in range(4):
                        val = auc_vals[ci] if ci < len(auc_vals) else np.nan
                        metrics["AUC"][ci].append(float(val))
                # etykieta: prefix_cohortIndex (np. 020_1)
                x_labels.append(f"{prefix}_{cohort_idx+1}")
    return metrics, x_labels


def read_comparison_metrics_list(metrics_folder, comparison_file_list):
    """
    For a list of tuples (filename, label) in comparison_file_list, tries to read each file from metrics_folder.
    Returns a list of elements: each element is a tuple (label, auc_values_list_of_4) or (label, None) if missing.
    """
    result = []
    for fname, label in comparison_file_list:
        path = os.path.join(metrics_folder, fname)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                val = df["AUC"].iloc[0]
                auc_list = eval(val) if isinstance(val, str) else list(val)
                auc_list = [float(x) for x in auc_list]
                result.append((label, auc_list))
            except Exception:
                result.append((label, None))
        else:
            result.append((label, None))
    return result

# DATA GATHERING
metrics_a, x_labels_a = gather_metrics_for_directories(directories)
metrics_b, x_labels_b = gather_metrics_for_directories(alternative_directories)

def make_labels(x_labels, max_length):
    """
    Funkcja do skracania etykiet na osi X do maksymalnej długości.
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    i = 0
    for x in x_labels:
        # x_labels[i] = x[:4] + str(i % max_length + 1)
        x_labels[i] = x[:4] + str(alphabet[i % max_length])
        i += 1
    return x_labels


x_labels_a = make_labels(x_labels_a, 15)
x_labels_b = make_labels(x_labels_b, 15)

if len(x_labels_a) != len(x_labels_b):
    print("Warning: Different label lengths between sets (A vs B).")

comparison_metrics_a = read_comparison_metrics_list(metrics_alt, comparison_files_a)
comparison_metrics_b = read_comparison_metrics_list(metrics_high, comparison_files_b)

fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

for ax_idx, (metrics_set, x_labels, ax, comparison_metrics, title) in enumerate([
    (metrics_a, x_labels_a, axes[0], comparison_metrics_a, "Alt"),
    (metrics_b, x_labels_b, axes[1], comparison_metrics_b, "HighSignal")
]):
    n_points = len(x_labels)
    base_x = np.arange(n_points)
    offset = 0.06
    markers = ["o", "s", "^", "D"]
    for ci in range(4):
        xs = base_x + (ci - 1.5) * offset
        ys = metrics_set["AUC"][ci]
        ax.scatter(xs, ys, c=class_colors[ci], s=16, label=classes[ci], marker=markers[ci])

    for ref_idx, (ref_label, ref_vals) in enumerate(comparison_metrics):
        if ref_vals is None:
            continue
        for ci in range(4):
            linestyle = "--" if ref_idx == 0 else "-"
            label = ref_label if ci == 0 else None
            ax.axhline(ref_vals[ci], color=class_colors[ci], linestyle=linestyle, linewidth=1.2, label=label)

    ax.set_title(title)
    ax.set_xlabel("Modele")
    if ax_idx == 0:
        ax.set_ylabel("AUC")
    ax.grid(True)

    ax.set_xticks(base_x)
    ax.set_xticklabels(x_labels, rotation=75, fontsize=6, ha='right', rotation_mode='anchor')

# Legend
all_handles = []
all_labels = []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    all_handles.extend(h)
    all_labels.extend(l)

by_label = {}
for handle, label in zip(all_handles, all_labels):
    if label and label not in by_label:
        by_label[label] = handle

legend = fig.legend(by_label.values(), by_label.keys(),
                    loc='lower center',
                    bbox_to_anchor=(0.5, -0.02),
                    ncol=6,
                    frameon=True,
                    fontsize=9)

fig.subplots_adjust(bottom=0.01)
plt.suptitle("Porównanie wartości AUC wszystkich klas poszczególnych zbiorów", y=0.98, weight='bold')
plt.tight_layout(rect=[0.08, 0.04, 1, 0.95])
# plt.subplots_adjust(wspace=0.15, hspace=0.15)

fig.savefig("auc_comparison_two_sets.png", dpi=300, bbox_inches='tight')
plt.show()