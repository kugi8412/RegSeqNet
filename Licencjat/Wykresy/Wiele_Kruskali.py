#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agregacja MCC dla 4 podzbiorów + testy Kruskal-Wallis i parowe Mann-Whitney z korekcją BH.
Output: out_report.txt oraz wykresy boxplot.
"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from typing import List, Dict, Sequence, Optional

base_folder = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/"

directories = [
    os.path.join(base_folder, "data", "test_output", "cohorts", f"c{i}", "alt")
    for i in range(1, 11)
]

# Groups definition
group_indices = [
    [0],
    [1, 2, 4, 5, 6],
    [3],
    [7, 8, 9]
]

groups_as_paths: Optional[List[List[str]]] = None
prefixes = ["020", "040", "060", "080"]
n_models_per_prefix = 5
classes = ["PA", "NA", "PI", "NI"]

# Output
out_report = "kruskal_mw_comparisons_report.txt"
out_png_folder = "mcc_boxplots_by_class"
os.makedirs(out_png_folder, exist_ok=True)

# Colours for boxplots
group_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def safe_parse_list(s):
    if s is None:
        return None
    if isinstance(s, (list, tuple, np.ndarray)):
        return [float(x) for x in s]
    if isinstance(s, str):
        s_clean = s.replace('np.float64(', '').replace(')', '')
        try:
            parsed = ast.literal_eval(s_clean)
            if isinstance(parsed, (list, tuple)):
                return [float(x) for x in parsed]
        except Exception:
            try:
                parts = [p.strip() for p in s_clean.strip(' []').split(',') if p.strip()]
                return [float(p) for p in parts]
            except Exception:
                return None
    return None

def read_mcc_from_metrics_file(metrics_csv_path: str) -> Optional[List[float]]:
    """
    Czyta plik CSV i próbuje wyciągnąć kolumnę 'MCC' zawierającą listę 4 wartości.
    """
    if not os.path.isfile(metrics_csv_path):
        return None
    try:
        df = pd.read_csv(metrics_csv_path)
    except Exception:
        return None
    cols = {c.strip().lower(): c for c in df.columns}
    if 'auc' in cols:
        val = df[cols['auc']].iloc[0]
        lst = safe_parse_list(val)
        if lst and len(lst) >= 4:
            return lst[:4]

    cls_cols = []
    for name in ['pa','na','pi','ni']:
        if name in cols:
            cls_cols.append(cols[name])
    if len(cls_cols) == 4:
        try:
            return [float(df[c].iloc[0]) for c in cls_cols]
        except Exception:
            pass

    first_row = df.iloc[0]
    numeric = []
    for v in first_row:
        try:
            numeric.append(float(v))
        except Exception:
            continue
        if len(numeric) >= 4:
            break
    if len(numeric) >= 4:
        return numeric[:4]
    return None

def build_model_metrics_paths_for_cohort(cohort_high_path: str) -> List[str]:
    """
    Dla podanego katalogu '<...>/cX/high' generuje typowe ścieżki do plików metrics:
    """
    paths = []
    for prefix in prefixes:
        for i in range(1, n_models_per_prefix + 1):
            model_dirname = f"high_{prefix}_{i}"
            metrics_path = os.path.join(cohort_high_path, model_dirname, 'metrics', f'best_{model_dirname}_metrics.csv')
            paths.append(metrics_path)
    return paths

def aggregate_groups_mcc(directories: List[str], group_indices: Optional[List[List[int]]] = None,
                         groups_as_paths: Optional[List[List[str]]] = None) -> Dict[str, List[List[float]]]:
    """
    Zwraca: dict group_name -> list_of_4_lists (po jednym liście MCC dla każdej klasy)
    group_name będą: 'G1','G2','G3','G4'
    """
    groups = {}
    if groups_as_paths is not None:
        if len(groups_as_paths) != 4:
            raise ValueError("groups_as_paths musi mieć 4 elementy (po jednej liście ścieżek).")
        list_of_groups = groups_as_paths
    else:
        if group_indices is None:
            raise ValueError("Trzeba podać group_indices lub groups_as_paths.")
        list_of_groups = []
        for idx_list in group_indices:
            group_paths = []
            for idx in idx_list:
                if idx < 0 or idx >= len(directories):
                    raise IndexError(f"Index {idx} spoza zakresu directories (len={len(directories)})")
                group_paths.append(directories[idx])
            list_of_groups.append(group_paths)

    for gi, grp_paths in enumerate(list_of_groups, start=1):
        lists_per_class = [[] for _ in range(4)]
        for cohort_high_path in grp_paths:
            model_metric_paths = build_model_metrics_paths_for_cohort(cohort_high_path)
            for metrics_path in model_metric_paths:
                mccs = read_mcc_from_metrics_file(metrics_path)
                if mccs is None:
                    continue

                for ci in range(4):
                    try:
                        val = float(mccs[ci])
                    except Exception:
                        continue

                    lists_per_class[ci].append(val)
        groups[f"G{gi}"] = lists_per_class
    return groups

# Report
def compare_groups_and_write_report(groups_dict: Dict[str, List[List[float]]], report_path: str):
    """
    Dla każdej klasy wykonuje KW i parowe MW + korekcję BH i zapisuje szczegółowy raport.
    Tworzy też boxploty per klasa (4 grupy obok siebie).
    """
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("Porównanie grup (Kruskal-Wallis + parowe Mann-Whitney z korekcją Benjamina-Hochberga)\n")
        fh.write("Grupy: " + ", ".join(groups_dict.keys()) + "\n\n")
        group_names = list(groups_dict.keys())
        for ci, cls in enumerate(classes):
            fh.write(f"=== KLASA: {cls} ===\n")
            arrays = []
            for g in group_names:
                arr = np.array(groups_dict[g][ci], dtype=float)
                arr = arr[~np.isnan(arr)]
                arrays.append(arr)
                fh.write(f"{g}: n={len(arr)}, median={np.median(arr) if len(arr)>0 else 'NA'}\n")
            fh.write("\n")

            # Kruskal-Wallis
            nonempty_groups = sum(1 for a in arrays if len(a) > 0)
            if nonempty_groups < 2:
                fh.write("Brak wystarczających niepustych grup do testu Kruskal-Wallisa (min 2 wymagane).\n\n")
                continue
            try:
                arrays_for_kw = [a for a in arrays if len(a) > 0]
                stat_kw, p_kw = kruskal(*arrays_for_kw)
                fh.write(f"Kruskal-Wallis: H={stat_kw:.4f}, p={p_kw:.6e}\n")
            except Exception as e:
                fh.write(f"Kruskal-Wallis: błąd wykonania testu: {e}\n\n")
                continue

            # Test Mann-Whitney
            pairs = list(combinations(range(len(group_names)), 2))
            raw_pvals = []
            pair_info = []
            for i, j in pairs:
                a1 = arrays[i]
                a2 = arrays[j]
                if len(a1) < 2 or len(a2) < 2:
                    raw_pvals.append(np.nan)
                    pair_info.append((i, j, np.nan, np.nan, "insufficient data"))
                    continue
                try:
                    st, p = mannwhitneyu(a1, a2, alternative='two-sided')
                except Exception:
                    st, p = np.nan, np.nan
                raw_pvals.append(p)
                pair_info.append((i, j, st, p, "ok"))

            # correct B-H
            pvals_for_corr = [pv if not np.isnan(pv) else 1.0 for pv in raw_pvals]
            reject, pvals_bh, _, _ = multipletests(pvals_for_corr, alpha=0.05, method='fdr_bh')

            fh.write("Pair\tstat\tp_raw\tp_BH\treject_H0 (BH)\n")
            for idx_pair, (i, j, st, p, status) in enumerate(pair_info):
                gn_i = group_names[i]
                gn_j = group_names[j]
                pair_label = f"{gn_i} vs {gn_j}"
                if np.isnan(p):
                    fh.write(f"{pair_label}\t-\tNA\tNA\tinsufficient data\n")
                else:
                    fh.write(f"{pair_label}\t{st:.4f}\t{p:.6e}\t{pvals_bh[idx_pair]:.6e}\t{bool(reject[idx_pair])}\n")
            fh.write("\n")
            fh.write(f"Kruskal-Wallis result: p = {p_kw:.6e} -> {'istotny (p<0.05)' if p_kw < 0.05 else 'nieistotny (p>=0.05)'}\n")
            if p_kw >= 0.05:
                fh.write("Uwaga: Kruskal-Wallis nie wykazał istotnej różnicy między grupami; interpretacja parowych testów ograniczona.\n")
            fh.write("\n" + ("-"*80) + "\n\n")
    print(f"Zapisano raport: {report_path}")

    # Boxploty
    for ci, cls in enumerate(classes):
        fig, ax = plt.subplots(figsize=(8, 6))
        data_to_plot = [np.array(groups_dict[g][ci], dtype=float)[~np.isnan(np.array(groups_dict[g][ci], dtype=float))] for g in group_names]
        bp = ax.boxplot(data_to_plot, patch_artist=True, labels=group_names, showmeans=True)

        for patch, color in zip(bp['boxes'], group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor('black')
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.2)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_title(f"{cls} — rozkład MCC w 4 podgrupach")
        ax.set_ylabel("MCC")
        y_min, y_max = ax.get_ylim()
        y_text = y_min + 0.05 * (y_max - y_min)
        for i, arr in enumerate(data_to_plot):
            n = len(arr)
            med = np.median(arr) if n>0 else np.nan
            ax.text(i+1, y_text, f"n={n}\nmed={med:.3f}" if n>0 else "n=0\nmed=NA", ha='center', va='bottom', fontsize=8)
        out_png = os.path.join(out_png_folder, f"mcc_boxplot_{cls}.png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Zapisano wykres: {out_png}")

# Main
if __name__ == "__main__":
    if groups_as_paths is not None:
        groups_paths = groups_as_paths
    else:
        groups_paths = []
        for idx_list in group_indices:
            gp = []
            for idx in idx_list:
                gp.append(directories[idx])
            groups_paths.append(gp)

    # Aggregate MCC
    groups_mcc = aggregate_groups_mcc(directories, group_indices=group_indices, groups_as_paths=None)
    compare_groups_and_write_report(groups_mcc, out_report)
