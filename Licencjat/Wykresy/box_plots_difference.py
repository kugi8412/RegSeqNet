#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from scipy.stats import kruskal, mannwhitneyu
from collections import defaultdict
from matplotlib.patches import Patch
import csv
from statsmodels.stats.multitest import multipletests
from typing import Optional


def read_spatial_metrics(csv_file: str, column: str = 'AUC') -> Dict[str, List[float]]:
    """
    Czytanie CSV z metrykami przestrzennymi.
    """

    # Read the CSV into a single-row DataFrame
    df = pd.read_csv(csv_file)
    
    # Regex
    float_re = re.compile(r"[+-]?\d+\.\d+(?:[eE][+-]?\d+)?")

    metrics: Dict[str, List[float]] = {}

    # There should be exactly one row
    row = df.iloc[0]
    
    for col in df.columns:
        cell = str(row[col])
        nums = float_re.findall(cell)
        values = [float(x) for x in nums]
        metrics[col] = values
    
    return metrics[column]


def parse_impact_file(path: str) -> Dict[int, List[float]]:
    """
    Parsuje impact_best_<cohort>.txt:
    """
    data = {}
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    idx = None
    for ln in lines:
        m = re.match(r'Filter\s+(\d+):', ln)
        if m:
            idx = int(m.group(1))
            data[idx] = []
            continue
        m2 = re.match(r'Class\s+\d+:\s+AUC=([0-9\.]+),', ln)
        if m2 and idx is not None:
            data[idx].append(float(m2.group(1)))
    return data


def aggregate_diffs(
    base_dir: str,
    cohorts: List[str]
) -> Dict[int, Dict[int, List[float]]]:
    """
    Dla każdej ścieżki cohort w formacie '/c8/procent040_1', '/c9/procent060_2', ...
    (łączymy base_dir + cohort):
      - baseline: base_dir/high/high_<procent...>/metrics/best_high_<procent...>_metrics.csv
      - impact:   base_dir<cohort>/impact_best_<procent...>.txt
    Zwraca: {filter_idx: {class_idx: [baseline–impact diffs]}}
    """
    diffs: Dict[int, Dict[int, List[float]]] = {}
    for cohort in cohorts:
        # path
        cohort_path = base_dir + cohort

        # baseline metrics
        pct = cohort.split('/')[-1]
        pct_short = pct.replace('procent', '')
        c = cohort.split('/')[1]
        pct_short = pct_short.replace(c+'/', '')
        high_dir = os.path.join(base_dir, c, 'high', f'high_{pct_short}')
        metrics_csv = os.path.join(high_dir, 'metrics', f'best_high_{pct_short}_metrics.csv')
        if not os.path.isfile(metrics_csv):
            print(f"[WARN] missing baseline metrics for {cohort}: {metrics_csv}")
            continue
        baseline = read_spatial_metrics(metrics_csv)

        # impact file
        impact_txt = os.path.join(cohort_path, f'impact_best_{pct}.txt')
        if not os.path.isfile(impact_txt):
            print(f"[WARN] missing impact file for {cohort}: {impact_txt}")
            continue
        parsed = parse_impact_file(impact_txt)

        for idx, aucs in parsed.items():
            if len(aucs) != 4:
                continue

            if idx not in diffs:
                diffs[idx] = {cls: [] for cls in range(4)}

            for cls in range(4):
                diffs[idx][cls].append(baseline[cls] - aucs[cls])

    return diffs


def plot_filter_class_boxplots(
    data: Dict[int, Dict[int, List[float]]],
    selected_filters: List[int] = None,
    output_path: str = 'class_diffs_boxplots.png'
):
    """
    Rysuje 4 sub-wykresy (klasy 0-3):
      - Każdy boxplot pokazuje rozkład AUC
      - Jeśli selected_filters podane, rysujemy tylko je i w tej kolejności
      - Jeśli None, domyślnie sortujemy po medianie i bierzemy wszystkie
    """
    classes = list(next(iter(data.values())).keys())
    if selected_filters is None:
        # sort
        medians = {idx: np.median([d for cls_vals in cls_dict.values() for d in cls_vals])
                   for idx, cls_dict in data.items()}
        selected_filters = sorted(medians, key=lambda i: medians[i], reverse=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    KLASY = ['Promotor Aktywny', 'Niepromotor Aktywny', 'Promotor Nieaktywny', 'Niepromotor Nieaktywny']
    for ax, cls in zip(axes.flatten(), classes):
        vals = [data[filt][cls] for filt in selected_filters if filt in data]
        labels = [str(filt) for filt in selected_filters if filt in data]
        box = ax.boxplot(vals, tick_labels=labels, patch_artist=True)
        for b in box['boxes']:
            b.set_facecolor('blue')
            b.set_alpha(0.75)

        ax.set_title(f'{KLASY[cls]}')
        ax.set_xlabel('Indeksy filtrów', fontsize=12)
        ax.set_ylabel('% zmiana AUC', fontsize=12)
        ax.tick_params(axis='x', rotation=90)

    fig.suptitle('% zmiana AUC filtrów w zależności od klasy', fontsize=16, weight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def aggregate_max_pct_per_filter(
    base_dir: str,
    cohorts: List[str],
    selected_filters: List[int]
) -> Dict[int, List[float]]:
    """
    Zbiera dla każdego filtrow z selected_filters listy max % drop:
      (baseline_c - impact_c)/baseline_c*100, max po klasach
    """
    result = defaultdict(list)
    for cohort in cohorts:
        pct = cohort.split('/')[-1]
        pct_short = pct.replace('procent','')
        c = cohort.split('/')[1]
        # baseline path
        metrics_csv = os.path.join(base_dir, c, 'high',
                                   f'high_{pct_short}', 'metrics',
                                   f'best_high_{pct_short}_metrics.csv')
        if not os.path.isfile(metrics_csv):
            continue

        baseline = read_spatial_metrics(metrics_csv)
        impact_txt = os.path.join(base_dir + cohort, f'impact_best_{pct}.txt')
        if not os.path.isfile(impact_txt):
            continue
        impact = parse_impact_file(impact_txt)
        for filt in selected_filters:
            if filt not in impact:
                continue
            i_aucs = impact[filt]
            pct_diffs = [(baseline[k] - i_aucs[k]) / baseline[k] * 100
                         for k in range(4) if baseline[k] != 0]
            if pct_diffs:
                result[filt].append(max(pct_diffs))

    return result

def plot_selected_filters_boxplots(
    data: Dict[int, List[float]],
    output_path: str = 'selected_filters_max_pct.png',
    split_sizes: Optional[List[int]] = None,
    panel_colors: Optional[List[str]] = None,
    show_points: bool = False,
    max_xticks: int = 300
) -> None:
    """
    Rysuje 4 pionowo ułożone boxploty dla podzbiorów filtrów z "data".
    - data: dict {filter_idx: [values,...]} - kolejność według iteracji po data.keys()
    - split_sizes: opcjonalnie lista 4 liczb: ile filtrów na kolejne panele [1,2,3,4].
    - panel_colors: opcjonalna lista 4 kolorów (np. ['C0','C1','C2','C3'])
    - show_points: jeśli True, rysuje też pojedyncze obserwacje jako scatter
    - max_xticks: maksymalna liczba etykiet X do pokazania (reszta jest pomijana dla czytelności)
    Zapisuje wykres do output_path.
    """
    # ordered
    filters = list(data.keys())
    values = [data[f] for f in filters]
    n = len(filters)

    if panel_colors is None:
        panel_colors = ['#00FF00','c','#EF4026','#FF00FF']

    if split_sizes is None:
        if n <= 7:
            split_sizes = [n, 0, 0, 0]
        else:
            first = 7
            second = 13 if n > 7+13 else max(0, n-first)
            remaining = max(0, n - first - second)
            third = remaining // 2
            fourth = remaining - third
            split_sizes = [first, second, third, fourth]
    else:
        if len(split_sizes) != 4:
            raise ValueError("split_sizes musi mieć dokładnie 4 elementy (liczby filtrów dla paneli).")
        total_req = sum(split_sizes)
        if total_req > n:
            extra = total_req - n
            for i in range(3, -1, -1):
                dec = min(extra, split_sizes[i])
                split_sizes[i] -= dec
                extra -= dec
                if extra == 0:
                    break
        elif total_req < n:
            split_sizes[-1] += (n - total_req)

    splits = []
    idx = 0
    for sz in split_sizes:
        splits.append((idx, idx + sz))
        idx += sz

    # Plot
    n_panels = 4
    fig, axes = plt.subplots(n_panels, 1, figsize=(max(10, n/10), 4*n_panels), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for i, (start, end) in enumerate(splits):
        ax = axes[i]
        if start >= end:
            ax.axis('off')
            ax.set_title(f"Panel {i+1}: brak filtrów", fontsize=10)
            continue
        panel_filters = filters[start:end]
        panel_values = [data[f] for f in panel_filters]
        print(min(panel_values))
        x = np.arange(len(panel_filters))

        # boxplot
        bp = ax.boxplot(panel_values, positions=x, patch_artist=True, widths=0.6, showfliers=True)
        color = panel_colors[i % len(panel_colors)]
        for box in bp['boxes']:
            box.set(facecolor=color, alpha=0.7, edgecolor='black', )
        for median in bp.get('medians', []):
            median.set(color='black', linewidth=1.2)

        # scatter for points (optionally)
        if show_points:
            for xi, vals in enumerate(panel_values):
                if vals:
                    jitter = (np.random.rand(len(vals)) - 0.5) * 0.6
                    ax.scatter(np.full(len(vals), xi) + jitter, vals, color='k', s=6, alpha=0.6)

        # label X
        n_labels = len(panel_filters)
        if n_labels <= max_xticks:
            tick_positions = np.arange(n_labels)
            tick_labels = [str(f) for f in panel_filters]
        else:
            step = max(1, n_labels // max_xticks)
            tick_positions = np.arange(0, n_labels, step)
            tick_labels = [str(panel_filters[j]) for j in tick_positions]

        ax.set_xticks(tick_positions)
        if i == n_panels - 1:
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
        else:
            ax.set_xticklabels(tick_labels, rotation=75, fontsize=12)
        ax.set_xlim(-0.5, max(0.5, len(panel_filters)-0.5))
        ax.set_ylabel('Maksymalny % różnica AUC ', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_title(f"{i + 1} Grupa zawierająca {len(panel_filters)} filtrów", fontsize=11)

    plt.suptitle('Maksymalny % różnica AUC dla pojedynczej klasy po wyzerowaniu filtra', fontsize=14, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def load_filters(filepath: str) -> np.ndarray:
    """
    Reads a filter file and returns an array of shape (num_filters, 4, 19).
    """
    filters = []
    with open(filepath, 'r') as f:
        current = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>filter_'):
                if current:
                    filters.append(np.array(current, dtype=float))
                current = []
            else:
                row = [float(x) for x in line.split()]
                current.append(row)
        if current:
            filters.append(np.array(current, dtype=float))
    return np.stack(filters)

def plot_sorted_frobenius_norms(
    filters: np.ndarray,
    num_filters_to_show: int = None,
    output_path: str = 'filter_norms.png'
):
    """
    Plots Frobenius norms of filters sorted descending.
    
    :param filters: np.ndarray of shape (N,4,19)
    :param num_filters_to_show: how many top filters to display (by norm). If None, show all.
    """
    # compute norms
    norms = np.linalg.norm(filters.reshape(filters.shape[0], -1), axis=1)
    sorted_idx = np.argsort(norms)[::-1][:46]
    if num_filters_to_show is not None:
        sorted_idx = sorted_idx[:num_filters_to_show]
    sorted_norms = norms[sorted_idx]
    
    # x positions sequentially for sorted filters
    x = np.arange(len(sorted_idx))

    plt.figure(figsize=(12, 6))
    plt.bar(x, sorted_norms, color='c', alpha=0.75)
    plt.xlabel('Indeksy filtrów z pierwszej warstwy konwolucyjnej modelu high_signal4', fontsize=12)
    plt.ylabel('Norma Frobeniusa', fontsize=12)
    plt.title('Posortowane filtry po normie Frobeniusa', fontsize=16, weight='bold')
    plt.xticks(x, sorted_idx, rotation=90, fontsize=8)
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_avg_norms_across_files(
    filepaths: List[str],
    num_filters_to_show: int = None,
    output_path: str = 'avg_filter_norms.png'
):
    """
    Dla listy plików:
      - Wczytuje filtry, liczy normy (N, 4, 19) -> (N, )
      - Układa macierz norms_all: shape = (len(filepaths), N)
      - Oblicza średnią per filtra: avg_norms (N, )
      - Sortuje malejąco według avg_norms i wybiera top num_filters_to_show
      - Rysuje barplot
    """
    norms_list = []
    for path in filepaths:
        filters = load_filters(path)
        norms = np.linalg.norm(filters.reshape(filters.shape[0], -1), ord='nuc', axis=1)
        norms_list.append(norms)
    norms_all = np.vstack(norms_list)  # shape (F, N)
    
    avg_norms = norms_all.mean(axis=0)  # shape (N,)

    sorted_idx = np.argsort(avg_norms)[::-1]
    if num_filters_to_show is not None:
        sorted_idx = sorted_idx[:num_filters_to_show]
    sorted_avg = avg_norms[sorted_idx]
    
    # Plot
    x = np.arange(len(sorted_idx))
    plt.figure(figsize=(max(12, len(x)*0.1), 6))
    plt.bar(x, sorted_avg, color='c', alpha=0.8)
    plt.xlabel('Posortowane filtry po średniej', fontsize=12)
    plt.ylabel('Średnia norma Frobeniusa', fontsize=12)
    plt.title('Średnia norma Frobeniusa wszystkich modeli', fontsize=16, weight='bold')
    plt.xticks(x, sorted_idx, rotation=90, fontsize=8)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_avg_norms_ignore_low_norms(
    filepaths: List[str],
    norm_threshold: float,
    top_k: int = None,
    output_path: str = 'avg_norms.png'
):
    norms = []
    for p in filepaths:
        f = load_filters(p)
        n = np.linalg.norm(f.reshape(f.shape[0], -1), axis=1)
        norms.append(n)
    norms = np.stack(norms, axis=0)  # shape (F, N)

    avg = []
    for col in norms.T:
        valid = col > norm_threshold
        if valid.any():
            avg.append(col[valid].mean())
        else:
            avg.append(np.nan)
    avg = np.array(avg)

    order = np.argsort(avg)[::-1]
    if top_k is not None:
        order = order[:top_k]
    vals = avg[order]

    start = 100
    end = 150
    x = np.arange(len(order))
    plt.figure(figsize=(12, 6))
    plt.bar(x[start:end], vals[start:end], color='c', alpha=0.75)
    plt.xticks(x[start:end], order[start:end], rotation=90, fontsize=12)
    plt.xlabel('Posortowane malejąca indeksy filtrów', fontsize=12)
    plt.ylabel(f'Średnia norma > {norm_threshold}')
    plt.title('Średnia norma Frobeniusa dla wszystkich filtrów powyżej progu', fontsize=16, weight='bold')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def aggregate_extreme_pct_per_filter(
    base_dir: str,
    cohorts: List[str],
    selected_filters: List[int]
) -> Dict[int, Dict[str, List[float]]]:
    """
    Zbiera dla każdego filtra in selected_filters:
      - lista maksymalnych % spadków AUC (max po klasach)
      - lista minimalnych % spadków AUC (min po klasach)
    Zwraca: {filter_idx: {'max': [...], 'min': [...]}}
    """
    result = {filt:{'max':[], 'min':[]} for filt in selected_filters}
    for cohort in cohorts:
        pct = cohort.split('/')[-1]
        pct_short = pct.replace('procent','')
        c = cohort.split('/')[1]
        # baseline
        metrics_csv = os.path.join(base_dir, c, 'high',
                                   f'high_{pct_short}','metrics',
                                   f'best_high_{pct_short}_metrics.csv')
        if not os.path.isfile(metrics_csv):
            continue
        baseline = read_spatial_metrics(metrics_csv)
        # impact
        impact_txt = os.path.join(base_dir + cohort, f'impact_best_{pct}.txt')
        if not os.path.isfile(impact_txt):
            continue
        impact = parse_impact_file(impact_txt)

        for filt in selected_filters:
            if filt not in impact:
                continue
            aucs = impact[filt]
            if len(aucs)!=4:
                continue
            pct_diffs = [ (baseline[k]-aucs[k])/baseline[k]*100 
                          for k in range(4) if baseline[k]!=0 ]
            if not pct_diffs:
                continue
            result[filt]['max'].append(max(pct_diffs))
            result[filt]['min'].append(min(pct_diffs))
    return result

def plot_extreme_boxplots_panels(
    data: Dict[int, Dict[str, List[float]]],
    output_path: str = 'extreme_pct_drops_panels.png',
    split_sizes: Optional[List[int]] = None,
    panel_colors: Optional[List[str]] = None,
    show_points: bool = False,
    max_xticks: int = 300
) -> None:
    """
    Rysuje 4 pionowo ułożone panele. W każdym panelu dla kolejnych filtrów rysuje parę boxplotów:
    - lewy = 'max' (lista wartości data[filter]['max'])
    - prawy = 'min' (lista wartości data[filter]['min'])

    Parametry:
      - data: dict {filter_idx: {'max': [...], 'min': [...]}}
      - output_path: ścieżka do zapisu PNG
      - split_sizes: opcjonalna lista 4 wartości: ile filtrów w kolejnych panelach.
      - panel_colors: lista 4 kolorów (kolor pudełek max i min będzie bazował na tej parze)
      - show_points: czy nanieść poszczególne pomiary jako scatter (rozdrobnienie)
      - max_xticks: maksymalna liczba etykiet X do pokazania w jednym panelu
    """
    # Order for dict
    filters = list(data.keys())
    n = len(filters)

    if panel_colors is None:
        panel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    if split_sizes is None:
        if n <= 7:
            split_sizes = [n, 0, 0, 0]
        else:
            first = 7
            second = 13 if n > 7+13 else max(0, n-first)
            remaining = max(0, n - first - second)
            third = remaining // 2
            fourth = remaining - third
            split_sizes = [first, second, third, fourth]
    else:
        if len(split_sizes) != 4:
            raise ValueError("split_sizes musi mieć dokładnie 4 elementy.")
        total_req = sum(split_sizes)
        if total_req > n:
            extra = total_req - n
            for i in range(3, -1, -1):
                dec = min(extra, split_sizes[i])
                split_sizes[i] -= dec
                extra -= dec
                if extra == 0:
                    break
        elif total_req < n:
            split_sizes[-1] += (n - total_req)

    splits = []
    idx = 0
    for sz in split_sizes:
        splits.append((idx, idx + sz))
        idx += sz

    n_panels = 4
    fig, axes = plt.subplots(n_panels, 1, figsize=(max(10, n/8), 4*n_panels), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for pi, (start, end) in enumerate(splits):
        ax = axes[pi]
        if start >= end:
            ax.axis('off')
            ax.set_title(f"Panel {pi+1}: brak filtrów", fontsize=10)
            continue

        panel_filters = filters[start:end]
        panel_max = []
        panel_min = []
        counts = []
        for f in panel_filters:
            vals_max = data.get(f, {}).get('max', []) or []
            vals_min = data.get(f, {}).get('min', []) or []
            if len(vals_max) == 0:
                panel_max.append([np.nan])
            else:
                panel_max.append(list(vals_max))
            if len(vals_min) == 0:
                panel_min.append([np.nan])
            else:
                panel_min.append(list(vals_min))
            counts.append((len(vals_max), len(vals_min)))

        m = len(panel_filters)
        idxs = np.arange(m)
        width = 0.35
        pos_max = idxs - width/2
        pos_min = idxs + width/2

        # Boxplots
        bmax = ax.boxplot(panel_max, positions=pos_max, widths=width,
                          patch_artist=True, showfliers=True,
                          flierprops=dict(marker='o', color='blue', alpha=0.6, markersize=4))
        bmin = ax.boxplot(panel_min, positions=pos_min, widths=width,
                          patch_artist=True, showfliers=True,
                          flierprops=dict(marker='^', color='red', alpha=0.6, markersize=4))

        # Colours
        color = panel_colors[pi % len(panel_colors)]
        import matplotlib.colors as mcolors
        try:
            light = mcolors.to_hex(mcolors.to_rgb(color) * 0.7 + np.array([0.3,0.3,0.3])*0.3)
        except Exception:
            light = color

        # style boxów
        for box in bmax['boxes']:
            box.set(facecolor=color, alpha=0.9, edgecolor='black')
        for box in bmin['boxes']:
            box.set(facecolor=light, alpha=0.45, edgecolor='black')

        for median in bmax.get('medians', []):
            median.set(color='black', linewidth=1.2)
        for median in bmin.get('medians', []):
            median.set(color='black', linewidth=1.2)

        if show_points:
            for xi, (vals_mx, vals_mn) in enumerate(zip(panel_max, panel_min)):
                if not (len(vals_mx) == 1 and np.isnan(vals_mx[0])):
                    jitter = (np.random.rand(len(vals_mx)) - 0.5) * (width*0.9)
                    ax.scatter(np.full(len(vals_mx), pos_max[xi]) + jitter, vals_mx,
                               color='black', s=6, alpha=0.6)
                if not (len(vals_mn) == 1 and np.isnan(vals_mn[0])):
                    jitter = (np.random.rand(len(vals_mn)) - 0.5) * (width*0.9)
                    ax.scatter(np.full(len(vals_mn), pos_min[xi]) + jitter, vals_mn,
                               color='gray', s=6, alpha=0.6)

        # xticks: limit labels
        n_labels = len(panel_filters)
        if n_labels <= max_xticks:
            tick_positions = idxs
            tick_labels = [str(f) for f in panel_filters]
        else:
            step = max(1, n_labels // max_xticks)
            tick_positions = np.arange(0, n_labels, step)
            tick_labels = [str(panel_filters[j]) for j in tick_positions]

        ax.set_xticks(tick_positions)
        if pi == n_panels - 1:
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
        else:
            ax.set_xticklabels(tick_labels, rotation=75, fontsize=12)
        ax.set_xlim(-0.8, max(0.8, m-0.2))
        ax.set_ylabel('Największa i najmniejsza % różnica AUC', fontsize=10)
        ax.set_title(f"{pi+1} Grupa zawierająca {len(panel_filters)} filtrów", fontsize=11)

        for xi, (cnt_m, cnt_n) in enumerate(counts):
            if cnt_m == 0 and cnt_n == 0:
                ax.text(xi, ax.get_ylim()[0] + 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                        "n=0", ha='center', va='bottom', fontsize=7, color='red')

        ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Legend
    import matplotlib.patches as mpatches
    max_patch = mpatches.Patch(facecolor=panel_colors[0] if panel_colors else "#21435c", edgecolor='black', label='Max % drop')
    min_patch = mpatches.Patch(facecolor='lightgray', edgecolor='black', label='Min % drop')
    # fig.legend(handles=[max_patch, min_patch], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02), frameon=True)

    plt.suptitle('Największy i najmniejszy % zmiana AUC dla wyzerowanych filtrów', fontsize=14, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_avg_spectral_norms_across_files(
    filepaths: List[str],
    num_filters_to_show: int = None,
    output_path: str = 'avg_spectral_norms.png'
):
    """
    Dla listy plików:
      - Wczytuje filtry, liczy normę spektralną (największa wartość singularna)
        każdego 4 x 19 filtra
      - Układa macierz norms_all: shape=(len(filepaths), N)
      - Oblicza średnią per filtra
      - Sortuje malejąco i wybiera top num_filters_to_show
      - Rysuje barplot: x = rank, etykiety = rzeczywiste numery filtrów
    """
    norms_list = []
    for path in filepaths:
        filters = load_filters(path)  # shape (N,4,19)
        # spectral norm per filter
        norms = np.array([np.linalg.norm(filt, ord=-2) for filt in filters])
        norms_list.append(norms)
    norms_all = np.vstack(norms_list)  # shape (F, N)
    
    # average per filter
    avg_norms = norms_all.mean(axis=0)
    
    # sort descending
    sorted_idx = np.argsort(avg_norms)[::-1]
    if num_filters_to_show is not None:
        sorted_idx = sorted_idx[:num_filters_to_show]
    sorted_avg = avg_norms[sorted_idx]
    
    # plot
    x = np.arange(len(sorted_idx))
    plt.figure(figsize=(max(12, len(x)*0.1), 6))
    plt.bar(x, sorted_avg, color='c', alpha=0.8)
    plt.xlabel('Filtry posortowane wg średniej normy spektralnej', fontsize=12)
    plt.ylabel('Średnia norma spektralna', fontsize=12)
    plt.title('Średnia norma spektralna wszystkich modeli', fontsize=16, weight='bold')
    plt.xticks(x, sorted_idx, rotation=90, fontsize=8)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def group_and_test(
    data: Dict[int, List[float]],
    output_prefix: str = 'filter_groups'
):
    """
    Groups filters into three groups: first 10, middle, last 10 (based on sorted keys order).
    Performs Kruskal-Wallis across the three groups and pairwise Mann-Whitney U tests.
    Saves results to a txt and plots boxplots.
    """
    # Sort filters by key
    print(data.keys())
    filters = list(data.keys())
    n = len(filters)
    # define groups
    group1 = filters[:10]
    group3 = filters[-10:]
    group2 = filters[10:-10] if n > 20 else []
    
    # gather values
    vals1 = [data[f] for f in group1]
    vals2 = [data[f] for f in group2] if group2 else []
    vals3 = [data[f] for f in group3]
    
    # flatten per group
    flat1 = np.concatenate(vals1) if vals1 else np.array([])
    flat2 = np.concatenate(vals2) if vals2 else np.array([])
    flat3 = np.concatenate(vals3) if vals3 else np.array([])
    
    # Kruskal-Wallis
    groups_for_kw = [g for g in [flat1, flat2, flat3] if len(g)>0]
    kw_stat, kw_p = kruskal(*groups_for_kw)
    
    # pairwise Mann-Whitney U
    mw_results = {}
    if len(flat1)>0 and len(flat2)>0:
        stat12, p12 = mannwhitneyu(flat1, flat2, alternative='two-sided')
        mw_results['1_vs_2'] = (stat12, p12)
    if len(flat1)>0 and len(flat3)>0:
        stat13, p13 = mannwhitneyu(flat1, flat3, alternative='two-sided')
        mw_results['1_vs_3'] = (stat13, p13)
    if len(flat2)>0 and len(flat3)>0:
        stat23, p23 = mannwhitneyu(flat2, flat3, alternative='two-sided')
        mw_results['2_vs_3'] = (stat23, p23)
    
    # save to txt
    txt = f"{output_prefix}_stats.txt"
    with open(txt, 'w') as f:
        f.write("Kruskal-Wallis test across 3 groups\n")
        f.write(f"Statistic: {kw_stat:.4f}, p-value: {kw_p:.4g}\n\n")
        f.write("Mann-Whitney U pairwise tests:\n")
        for k, v in mw_results.items():
            if isinstance(v, dict):
                f.write(f"{k}: U={v['stat']:.4f}, p-raw={v['p_raw']:.4g}, p-adj={v['p_adj']:.4g}, reject_H0={v['reject_H0']}\n")
            else:
                stat, p = v
                f.write(f"{k}: U={stat:.4f}, p-value={p:.4g}\n")
    print(f"Saved stats to {txt}")
    
    # plot boxplots
    fig, ax = plt.subplots(figsize=(8,6))
    box_data = [flat1, flat2, flat3]
    labels = ['First 10', 'Middle', 'Last 10']
    ax.boxplot(box_data, labels=labels, patch_artist=True)
    for patch, color in zip(ax.artists, ['#FF9999','#99FF99','#9999FF']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Max % AUC Drop by Filter Group")
    ax.set_ylabel("Max % drop")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{output_prefix}_boxplot.png", dpi=300)
    plt.close(fig)

def group_and_test_violin(
    data: Dict[int, List[float]],
    output_prefix: str = 'filter_groups'
):
    """
    Groups filters into three groups: first 10, middle, last 10.
    Performs Kruskal-Wallis and Mann-Whitney U tests, saves stats,
    and plots an aesthetic violin plot.
    """
    # Sort filters by key
    filters = list(data.keys())
    n = len(filters)
    # define groups
    # (0, 7), (7, 58 - 10), (58 - 10, 99), (99, 300)
    group1 = filters[0:7]
    group2 = filters[7:47]
    group3 = filters[47:90] if n > 20 else []
    group4 = filters[90:] if n > 100 else []

    
    # gather values
    vals1 = np.concatenate([data[f] for f in group1]) if group1 else np.array([])
    vals2 = np.concatenate([data[f] for f in group2]) if group2 else np.array([])
    vals3 = np.concatenate([data[f] for f in group3]) if group3 else np.array([])
    vals4 = np.concatenate([data[f] for f in group4]) if group4 else np.array([])
    
    # statistical tests
    groups_for_kw = [g for g in [vals1, vals2, vals3, vals4] if len(g)>0]
    kw_stat, kw_p = kruskal(*groups_for_kw)
    
    mw_results = {}
    pvals = []
    pairs = []

    if len(vals1) > 0 and len(vals2) > 0:
        stat12, p12 = mannwhitneyu(vals1, vals2, alternative='two-sided')
        mw_results['1_vs_2'] = (stat12, p12)
        pairs.append('1_vs_2')
        pvals.append(p12)

    if len(vals1) > 0 and len(vals3) > 0:
        stat13, p13 = mannwhitneyu(vals1, vals3, alternative='two-sided')
        mw_results['1_vs_3'] = (stat13, p13)
        pairs.append('1_vs_3')
        pvals.append(p13)

    if len(vals2) > 0 and len(vals3) > 0:
        stat23, p23 = mannwhitneyu(vals2, vals3, alternative='two-sided')
        mw_results['2_vs_3'] = (stat23, p23)
        pairs.append('2_vs_3')
        pvals.append(p23)

    if len(vals1) > 0 and len(vals4) > 0:
        stat14, p14 = mannwhitneyu(vals1, vals4, alternative='two-sided')
        mw_results['1_vs_4'] = (stat14, p14)
        pairs.append('1_vs_4')
        pvals.append(p14)

    if len(vals2) > 0 and len(vals4) > 0:
        stat24, p24 = mannwhitneyu(vals2, vals4, alternative='two-sided')
        mw_results['2_vs_4'] = (stat24, p24)
        pairs.append('2_vs_4')
        pvals.append(p24)

    if len(vals3) > 0 and len(vals4) > 0:
        stat34, p34 = mannwhitneyu(vals3, vals4, alternative='two-sided')
        mw_results['3_vs_4'] = (stat34, p34)
        pairs.append('3_vs_4')
        pvals.append(p34)

    # Benjamini-Hochberg correction
    if pvals:
        reject, pvals_corr, _, _ = multipletests(pvals, method='fdr_bh')
        # nadpisujemy wyniki skorygowanymi p-value
        for pair, adj_p, rej in zip(pairs, pvals_corr, reject):
            stat, orig_p = mw_results[pair]
            mw_results[pair] = {
                'stat': stat,
                'p_raw': orig_p,
                'p_adj': adj_p,
                'reject_H0': rej
            }

    # save to txt
    txt = f"{output_prefix}_stats.txt"
    with open(txt, 'w') as f:
        f.write("Kruskal-Wallis test across 3 groups\n")
        f.write(f"Statistic: {kw_stat:.4f}, p-value: {kw_p:.4g}\n\n")
        f.write("Mann-Whitney U pairwise tests:\n")
        for k, v in mw_results.items():
            if isinstance(v, dict):
                f.write(f"{k}: U={v['stat']:.4f}, p-raw={v['p_raw']:.4g}, p-adj={v['p_adj']:.4g}, reject_H0={v['reject_H0']}\n")
            else:
                stat, p = v
                f.write(f"{k}: U={stat:.4f}, p-value={p:.4g}\n")
    print(f"Saved stats to {txt}")
    # plot violin
    fig, ax = plt.subplots(figsize=(12,6))
    parts = ax.violinplot([vals1, vals2, vals3, vals4], positions=[1, 2, 3, 4],
                          showmeans=False, showmedians=False, showextrema=False)

    colors = ['#00FF00','c','#EF4026','#FF00FF']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.95)

    vals = [vals1, vals2, vals3, vals4]
    q1 = [np.percentile(v, 25) for v in vals]
    med = [np.percentile(v, 50) for v in vals]
    q3 = [np.percentile(v, 75) for v in vals]
    whiskers = [adjacent_values(v, a, b) for v, a, b in zip(vals, q1, q3)]
    lower = [w[0] for w in whiskers]
    upper = [w[1] for w in whiskers]

    # whiskers and style
    inds = [1, 2, 3, 4]
    ax.scatter(inds, med, marker='o', color='white', s=64, zorder=3)
    ax.vlines(inds, q1, q3, color='#580F41', lw=12)
    ax.vlines(inds, lower, upper, color='#580F41', lw=1)

    set_axis_style(ax, ['7 Filtrów o największej normie','Filtry zbliżone do modelu alt_last','Filtry grupy plateu','Filtry o najmniejszej normie'])
    ax.set_ylabel('Max % Δ AUC', fontsize=12)
    ax.set_title('Maksymalna różnica w AUC dla grup filtrów pierwszej warstwy', fontsize=16, weight='bold')
    ax.grid(axis='y', alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_prefix, dpi=300)
    plt.close(fig)
    # style means and medians
    """
    parts['cmeans'].set_edgecolor('k')
    parts['cmedians'].set_edgecolor('k')
    
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(['First 10','Middle','Last 10'], fontsize=10)
    ax.set_ylabel('Max % drop in AUC', fontsize=12)
    ax.set_title('Distribution of Max % AUC Drop by Filter Groups', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{output_prefix}_violin.png", dpi=300)
    plt.close(fig)"""

def adjacent_values(vals, q1, q3):
    upper = q3 + (q3 - q1) * 1.5
    upper = np.clip(upper, q3, max(vals))
    lower = q1 - (q3 - q1) * 1.5
    lower = np.clip(lower, min(vals), q1)
    return lower, upper

def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    # ax.set_xlabel(xlabel)

def compute_file_metrics(
    filters: np.ndarray,
    selected_indices: List[int]
) -> Dict[int, Dict[str, float]]:
    """
    For a single file's filters, compute for each idx in selected_indices:
      - min_fro_diff: minimal Frobenius norm difference to any other selected filter
      - min_angle: minimal cosine angle to any other selected filter
    """
    D = filters.shape[1] * filters.shape[2]
    # flatten
    vecs = {i: filters[i].reshape(-1) for i in selected_indices}
    norms = {i: np.linalg.norm(vecs[i]) for i in selected_indices}
    # normalize for angle
    unit = {i: vecs[i]/norms[i] if norms[i]>0 else vecs[i] for i in selected_indices}
    
    metrics = {}
    for i in selected_indices:
        min_diff = np.inf
        min_ang = np.inf
        for j in selected_indices:
            if i == j: continue
            # fro norm diff
            diff = abs(norms[i] - norms[j])
            if diff < min_diff:
                min_diff = diff
            # cos angle
            cos_sim = np.dot(unit[i], unit[j])
            cos_sim = np.clip(cos_sim, -1, 1)
            angle = np.arccos(cos_sim)
            if angle < min_ang:
                min_ang = angle
        metrics[i] = {'min_fro_diff': min_ang, 'min_angle': min_ang}
    return metrics

def aggregate_rank_across_files(
    filepaths: List[str],
    selected_indices: List[int]
):
    """
    For each file, compute metrics, derive ranks separately for min_fro_diff and min_angle
    (larger metric -> higher rank), then average ranks across files for each filter.
    """
    # store ranks per file
    ranks_fro = {i: [] for i in selected_indices}
    ranks_ang = {i: [] for i in selected_indices}
    
    for fp in filepaths:
        filters = load_filters(fp)
        m = compute_file_metrics(filters, selected_indices)
        fro_vals = np.array([m[i]['min_fro_diff'] for i in selected_indices])
        ang_vals = np.array([m[i]['min_angle'] for i in selected_indices])
        fro_order = np.argsort(-fro_vals)
        ang_order = np.argsort(-ang_vals)
        fro_ranks = np.empty_like(fro_order)
        fro_ranks[fro_order] = np.arange(1, len(selected_indices)+1)
        ang_ranks = np.empty_like(ang_order)
        ang_ranks[ang_order] = np.arange(1, len(selected_indices)+1)
        for idx, i in enumerate(selected_indices):
            ranks_fro[i].append(fro_ranks[idx])
            ranks_ang[i].append(ang_ranks[idx])
    
    # average ranks
    avg_ranks = {i: (np.mean(ranks_fro[i]) + np.mean(ranks_ang[i]))/2 for i in selected_indices}
    return avg_ranks

def plot_average_ranking(
    avg_ranks: Dict[int, float],
    output_path: str = 'average_filter_ranking.png'
):
    """
    Rysuje barplot filtrow posortowany rosnąco po średniej randze.
    """
    # Sort filters by ascending average rank
    sorted_items = sorted(avg_ranks.items(), key=lambda x: x[1])
    filters = [item[0] for item in sorted_items]
    ranks = [item[1] for item in sorted_items]

    x = np.arange(len(filters))
    plt.figure(figsize=(max(12, len(filters)*0.1), 6))
    plt.bar(x, ranks, color='c', alpha=0.8)
    plt.xticks(x, filters, rotation=90, fontsize=8)
    plt.xlabel('Filter index')
    plt.ylabel('Average rank')
    plt.title('Average Filter Ranking by Local Distinctiveness (sorted)', fontsize=14, weight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

import os
import re
from collections import Counter
from typing import List, Dict

def parse_impact_filters(path: str) -> List[int]:
    """
    Wczytuje plik impact i zwraca listę numerów filtrów, które w nim wystąpiły.
    """
    filters = []
    with open(path, 'r') as f:
        for line in f:
            m = re.match(r'Filter\s+(\d+):', line.strip())
            if m:
                filters.append(int(m.group(1)))
    return filters

def count_filter_occurrences(
    base_dir: str,
    cohorts: List[str]
) -> Dict[int, int]:
    """
    Przechodzi po zadanych kohortach, czyta każdy plik impact_best_*.txt
    i zlicza, ile razy pojawia się każdy filter_idx.
    """
    counter = Counter()
    for cohort in cohorts:
        cohort_path = os.path.join(base_dir, cohort)
        for fn in os.listdir(cohort_path):
            if fn.startswith('impact_best_') and fn.endswith('.txt'):
                path = os.path.join(cohort_path, fn)
                try:
                    flts = parse_impact_filters(path)
                    counter.update(flts)
                except Exception as e:
                    print(f"[WARN] Nie udało się przetworzyć {path}: {e}")
    return dict(counter)

def sorted_filter_counts(
    counts: Dict[int, int]
) -> List[tuple]:
    """
    Sortuje słownik filter_idx->count malejąco według count.
    Zwraca listę krotek (filter_idx, count).
    """
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)

def plot_top_filter_counts(
    sorted_counts: List[tuple],
    top_n: int = 20,
    output_path: str = 'filter_occurrence_counts.png'
):
    """
    Rysuje słupkowy wykres top_n filtrów według liczby wystąpień.
    """
    # wybierz top_n
    top = sorted_counts[:top_n]
    idxs = [str(idx) for idx, _ in top]
    counts = [cnt for _, cnt in top]

    plt.figure(figsize=(max(10, top_n*0.4), 6))
    bars = plt.bar(range(top_n), counts, color='skyblue', edgecolor='black')
    
    plt.xticks(range(top_n), idxs, rotation=90, fontsize=8)
    plt.xlabel('Numer filtra', fontsize=12)
    plt.ylabel('Liczba wystąpień', fontsize=12)
    plt.title(f'Top {top_n} najczęściej zerowanych filtrów', fontsize=14, weight='bold')
    plt.grid(axis='y', alpha=0.3)

    # adding labels on top of bars
    for bar, cnt in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1, str(cnt),
                 ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_three_group_histograms(
    sorted_counts: List[Tuple[int, int]],
    output_path: str = 'three_group_histograms.png'
):
    """
    Podziel posortowane filtry na trzy równoliczne grupy i narysuj 3 histogramy obok siebie.
    Pierwszą grupę zaznacz w pierwszym podwykresie wyróżnionym kolorem.
    """
    total = len(sorted_counts)
    size = total // 3
    groups = [
        sorted_counts[0:size],
        sorted_counts[size:2*size],
        sorted_counts[2*size:]
    ]
    titles = ['Grupa 1 (najczęściej)', 'Grupa 2 (średnie)', 'Grupa 3 (najrzadziej)']
    colors = ['salmon', 'lightgray', 'lightgray']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, grp, title, color in zip(axes, groups, titles, colors):
        idxs = [str(idx) for idx, _ in grp]
        counts = [cnt for _, cnt in grp]
        x = np.arange(len(grp))
        bars = ax.bar(x, counts, color=color, edgecolor='black', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(idxs, rotation=90, fontsize=6)
        ax.set_title(title)
        ax.set_xlabel('Numer filtra')
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('Liczba wystąpień')
    fig.suptitle('Częstotliwość zerowania filtrów w 3 grupach', fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_three_group_histograms_vertical(
    sorted_counts: List[Tuple[int, int]],
    output_path: str = 'three_group_histograms_vertical.png'
):
    """
    Podziel posortowane filtry na trzy grupy (różne rozmiary) i narysuj 3 histogramy pionowo.
    Pierwsza grupa z większym odstępem między słupkami i wyróżniona kolorem.
    Dodana legenda oraz naturalne ticki na osi Y (co 10 w grupie 1, co 1 w pozostałych).
    """
    total = len(sorted_counts)
    size1 = 46
    size2 = 46 + 25 + 21 
    size3 = total - size1 - size2
    groups = [
        sorted_counts[0:size1],
        sorted_counts[size1:size1+size2],
        sorted_counts[size1+size2:]
    ]
    titles = ['Filtry o niezmienionych wartościach', 'Filtry o średniej częstotliwości', 'Filtry o rzadkiej częstotliwości'] 
    colors = ['salmon', 'c', 'c']
    spacings = [1.5, 1.5, 1.0]
    widths = [0.8, 0.8, 0.6]
    sizes = [8, 8, 7]

    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=False)
    for idx, (ax, grp, title, color, spacing) in enumerate(zip(axes, groups, titles, colors, spacings)):
        idxs = [str(i) for i, _ in grp]
        counts = [c for _, c in grp]
        x = np.arange(len(grp)) * spacing
        bars = ax.bar(x, counts, color=color, edgecolor='black', alpha=0.8, width=widths[idx])

        ax.set_xticks(x)
        ax.set_xticklabels(idxs, rotation=90, fontsize=sizes[idx])
        ax.set_title(title, fontsize=12)
        ax.set_ylabel('Liczba wystąpień', fontsize=12)
        ax.grid(axis='y', alpha=0.5)

        max_c = max(counts) if counts else 0
        if idx == 0:
            yticks = np.arange(0, max_c + 10, 10)
        else:
            yticks = np.arange(0, max_c + 1, 1)
        ax.set_yticks(yticks)

        # Labels on bars for counts between 30 and 60
        for bar, cnt in zip(bars, counts):
            if cnt > 30 and cnt < 60:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2,
                        height + max_c * 0.01,
                        str(cnt), ha='center', va='bottom', fontsize=8)

    axes[-1].set_xlabel('Indeks filtra', fontsize=12)
    fig.suptitle('Częstotliwość występowania filtrów o normach Frobeniusa > 0.04', fontsize=16, weight='bold')

    # Add legend
    legend_handles = [
        Patch(facecolor='salmon', edgecolor='black', label='Oryginalne filtry'),
        Patch(facecolor='c', edgecolor='black', label='Nowe filtry')
    ]
    fig.legend(handles=legend_handles, loc='upper right', title='Grupy filtrów')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()

_FILTER_RX = re.compile(r"filter[_\-\s]?(\d+)$", flags=re.IGNORECASE)
_DIGIT_RX = re.compile(r"(\d+)")

def parse_F(path: str) -> List[int]:
    """
    Wczytuje plik CSV o strukturze zawierającej kolumnę z nazwą filtra (np. "Filter")
    i zwraca listę indeksów filtrów (int) występujących w pliku.
    Toleruje formaty: 'filter_3', 'filter-3', 'Filter3', '3', itp.
    Pomija wiersze, których nie da się sparsować.
    """
    indices: List[int] = []
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Plik nie istnieje: {path}")

    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Plik '{path}' nie zawiera nagłówka CSV.")

        filter_col = None
        for c in reader.fieldnames:
            if c is None:
                continue
            if 'filter' in c.strip().lower():
                filter_col = c
                break

        if filter_col is None:
            cand = next((c for c in reader.fieldnames if c is not None), None)
            if cand is None:
                raise ValueError(f"Nie znaleziono kolumn do parsowania w pliku '{path}'")
            filter_col = cand

        for row in reader:
            raw = row.get(filter_col, "")
            if raw is None:
                continue
            raw = str(raw).strip()
            m = _FILTER_RX.search(raw)
            if m:
                idx = int(m.group(1))
                indices.append(idx)
                continue

            m2 = _DIGIT_RX.search(raw)
            if m2:
                idx = int(m2.group(1))
                indices.append(idx)
                continue

            print(f"[WARN] Nie udało się sparsować filtra z '{raw}' w pliku '{path}'")

    return indices


def count_Frobenius(
    base_dir: str,
    cohorts: List[str]
) -> Dict[int, int]:
    """
    Przechodzi po zadanych kohortach (każda kohorta to podfolder w base_dir),
    czyta pliki pasujące do wzorca 'best_high..._filters_above_tresh.csv'
    i zlicza, ile razy pojawia się każdy filter_idx.
    Zwraca słownik filter_idx -> count (liczba wystąpień we wszystkich plikach).
    """
    counter = Counter()
    for cohort in cohorts:
        cohort_path = os.path.join(base_dir, cohort)
        if not os.path.isdir(cohort_path):
            print(f"[WARN] Kohorta nie istnieje lub nie jest katalogiem: {cohort_path} -> pomijam")
            continue
        for fn in os.listdir(cohort_path):
            if fn.startswith('best_high') and fn.endswith('_filters_above_tresh.csv'):
                path = os.path.join(cohort_path, fn)
                try:
                    flts = parse_F(path)
                    counter.update(flts)
                except Exception as e:
                    print(f"[WARN] Nie udało się przetworzyć {path}: {e}")

    return dict(counter)

"""
if __name__ == '__main__':
    base_dir = 'C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/data/test_output/cohorts'
    cohorts = [
        f"{c}/procent{p}_{i}"
        for c in ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']
        for p in ['020','040','060','080']
        for i in range(1,6)
    ]
    cohorts = [
        f"{c}/stats"
        for c in ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']
    ]

    counts = count_Frobenius(base_dir, cohorts)
    sorted_counts = sorted_filter_counts(counts)

    # Wypisz top 20 najczęściej występujących filtrów
    print("Filter_idx  Count")
    for idx, cnt in sorted_counts[:100]:
        print(f"{idx:>10}  {cnt}")
    print(len(sorted_counts))

    # plot_top_filter_counts(sorted_counts, top_n=234, output_path='filter_counts.png')
    plot_three_group_histograms_vertical(sorted_counts, output_path='three_group_histograms_vertical.png')
"""


# Main script to analyze filter occurrences and plot results
if __name__ == '__main__':
    base_dir = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/data/test_output/cohorts"
    cohorts = [f"/{c}/procent{p}_{i}" for p in ['040','060','080','020'] for i in range(1,6) for c in ['c8','c9','c10']]
    files = []
    path = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/data/test_output/cohorts/"
    for c in ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']:
        files += [os.path.join(path, c + '/filter', f'best_high_{p}_{i}_filter.txt') for p in ['040','060','080','020'] for i in range(1,6)]
    
    files = [f for f in files if os.path.exists(f)]

    # plot_top_filter_counts(sorted_counts, top_n=234, output_path='filter_counts.png')
    FILTRY = ['205', '105', '153', '267', '229', '113', '30'] + ['104', '167', '203', '27', '103', '193', '138', '76', '216', '182', '65', '11', '292', '141', '53', '285', '136', '247', '23', '71', '70', '121', '288', '18', '158', '299', '191', '48', '188', '50', '108', '148', '221', '225', '214', '200', '159', '17', '106', '56', '236'] + ['74', '92', '194', '25', '239', '114', '22', '289', '174', '232', '255', '275', '177', '294', '35', '220', '146', '115', '254', '47', '263', '282', '149', '94', '21', '118', '102', '119', '64', '269', '222', '38', '245', '131', '268', '125', '68', '210', '257', '60', '150', '261', '155', '43', '7', '90', '127', '101', '16', '162', '129'] + ['251', '82', '171', '85', '12', '175', '84', '156', '164', '80', '287', '217', '204', '39', '278', '123', '120', '8', '187', '208', '212', '290', '296', '15', '166', '237', '186', '49', '137', '163', '219', '66', '96', '134', '62', '77', '24', '284', '297', '117', '5', '241', '196', '20', '93', '231', '88', '209', '67', '223', '178', '128', '262', '151', '274', '100', '140', '52', '109', '249', '28', '51', '259', '69', '32', '112', '273', '10', '34', '144', '29', '147', '199', '63', '195', '57', '266', '98', '160', '283', '169', '276', '54', '252', '227', '298', '279', '238', '59', '78', '83', '215', '133', '234', '0', '1', '26', '226', '192', '124', '246', '181', '110', '176', '248', '152', '281', '99', '165', '19', '14', '13', '132', '2', '184', '130', '270', '213', '86', '55', '242', '37', '256', '202', '89', '291', '272', '244', '180', '3', '157', '139', '168', '280', '122', '33', '277', '170', '295', '116', '260', '233', '154', '75', '253', '264', '230', '206', '135', '111', '228', '4', '240', '95', '79', '183', '45', '161', '189', '271', '87', '46', '44', '107', '58', '286', '185', '235', '293', '143']
    my_filters = [int(x) for x in FILTRY] # [205, 105, 267, 153, 30, 229, 76, 27, 104, 113, 167, 182, 191, 11, 71, 141, 53, 65, 285, 247, 188, 103, 292, 216, 138, 288, 50, 193, 48, 299, 148, 158, 17, 203, 18, 225, 121, 23, 221, 200, 108, 70, 187, 214, 236, 62, 64, 38, 140, 74, 69, 147, 177, 77, 284, 129, 274, 289, 28, 144, 263]
    diffs = aggregate_diffs(base_dir, cohorts)
    
    class_filter = ['205', '105', '153', '267', '229', '113', '30'] + ['104', '167', '203', '27', '103', '193', '138', '76', '216', '182', '65', '11', '292', '141', '53', '285', '136', '247', '23', '71', '70', '121', '288', '18', '158', '299', '191', '48', '188', '50', '108', '148', '221', '225', '214', '200', '159', '17', '236']
    class_filt = [int(x) for x in class_filter]

    plot_filter_class_boxplots(diffs, class_filt, output_path='class_diffs_boxplots.png')
    # print(read_spatial_metrics('C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/data/test_output/cohorts/c8/high/high_040_1/metrics/best_high_040_1_metrics.csv', 'AUC'))
    # print(parse_impact_file('C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/data/test_output/cohorts/c8/procent040_1/impact_best_procent040_1.txt'))
    filepath = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/filters/highsignal_last/high_signal4_last_filter.txt"
    filters = load_filters(filepath)
    plot_sorted_frobenius_norms(filters, output_path='filter_norms.png')
    
    # plot_avg_spectral_norms_across_files(files, num_filters_to_show=75, output_path='top50_avg_norms.png')
    # plot_avg_norms_ignore_low_norms(files, norm_threshold=0.1, top_k=300, output_path='avg_norms_gt_0.01.png')
    all_max = aggregate_max_pct_per_filter(base_dir, cohorts, selected_filters=my_filters)
    filters = list(all_max.keys())
    group1 = [f for f in filters if str(f) in FILTRY[0:7]]
    group2 = [f for f in filters if str(f) in FILTRY[7:48]]
    group3 = [f for f in filters if str(f) in FILTRY[48:99]] if 0 < 20 else []
    group4 = [f for f in filters if str(f) in FILTRY[99:]] if 0 < 100 else []
    print(len(group1), len(group2), len(group3), len(group4))

    new_all_max = defaultdict(list)
    klucze = set(all_max.keys())
    for f in FILTRY:
        if int(f) in klucze:
            new_all_max[int(f)] = all_max.get(int(f), [])


    group_and_test_violin(new_all_max, 'selected_filters')
    # filters = list(data.keys())
    # splits = [(0, 7), (7, 58 - 10), (58 - 10, 99), (99, 300)]
    plot_selected_filters_boxplots(new_all_max, 'filters_boxplots.png', split_sizes=[7, 40, 43, 144], show_points=False)
    data = aggregate_extreme_pct_per_filter(base_dir, cohorts, new_all_max.keys())
    plot_extreme_boxplots_panels(data, output_path='extreme_pct_drops.png', split_sizes=[7, 40, 43, 144])

    group1 = FILTRY[0:7]
    group2 = FILTRY[7:48]
    group3 = FILTRY[48:99]
    group4 = FILTRY[99:]

    # Convert to int for matching with all_max keys
    group1_int = set(map(int, group1))
    group2_int = set(map(int, group2))
    group3_int = set(map(int, group3))
    group4_int = set(map(int, group4))

    # Count how many elements from each group are in all_max
    all_max_keys = set(all_max.keys())
    count1 = len(group1_int & all_max_keys)
    count2 = len(group2_int & all_max_keys)
    count3 = len(group3_int & all_max_keys)
    count4 = len(group4_int & all_max_keys)

    print(f"Group 1: {count1} elements")
    print(f"Group 2: {count2} elements")
    print(f"Group 3: {count3} elements")
    print(f"Group 4: {count4} elements")
    # avg_ranks = aggregate_rank_across_files(files, my_filters)
    # plot_average_ranking(avg_ranks)
