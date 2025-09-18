#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
from collections import defaultdict
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CONFIG
group_indices = [
    [7, 8, 9],
    [3],
    [0],
    [1, 2, 4, 5, 6]
]

folder_path = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/"

directories = [
    folder_path + "data/test_output/cohorts/c1/stats/",
    folder_path + "data/test_output/cohorts/c2/stats/",
    folder_path + "data/test_output/cohorts/c3/stats/",
    folder_path + "data/test_output/cohorts/c4/stats/",
    folder_path + "data/test_output/cohorts/c5/stats/",
    folder_path + "data/test_output/cohorts/c6/stats/",
    folder_path + "data/test_output/cohorts/c7/stats/",
    folder_path + "data/test_output/cohorts/c8/stats/",
    folder_path + "data/test_output/cohorts/c9/stats/",
    folder_path + "data/test_output/cohorts/c10/stats/",
]

# Data
new_files = [
    folder_path + 'data/test_output/cohorts/reference/origin000_highsignal_filter_above_tresh.csv',
    folder_path + 'data/test_output/cohorts/reference/best000_highsignal_filter_above_thresh.csv',
    folder_path + 'data/test_output/cohorts/reference/best0100_highsignal_filter_above_tresh.csv',
    folder_path + 'data/test_output/cohorts/reference/best0100_highsignal_filter_above_tresh.csv',
]

reference_file = folder_path + 'statistics/Filtersplot/origin000_highsignal_filters_above_tresh.csv'

out_csv = "model_filters_summary_newonly_frobenius.csv"
out_png = "models_newshading_only_pages.png"

# Frobenius threshold
TH_LOW = 0.04
TH_HIGH = 0.5

# Plot - params
page_size = 100
max_models = 202
max_xticks = 100

# regex
RE_MODEL = re.compile(r"(?:best[_\-a-z0-9]*?_)?(?:high[_\-]?)?(\d{1,3})[_\-]?(\d+)_filters", flags=re.I)
RE_MODEL_FALLBACK = re.compile(r"best.*?(\d{1,3})[_\-]?(\d+)", flags=re.I)

# Normalize files name
def normalize_filter_name(name: str) -> str:
    if name is None:
        return ""
    s = str(name).strip().lower()
    s = re.sub(r'^[`"\']+|[`"\']+$', '', s)
    digits = re.findall(r'(\d+)', s)
    if re.search(r'filter|filt', s):
        if digits:
            return f"filter_{int(digits[0])}"
        else:
            s2 = re.sub(r'[^a-z0-9]+', '_', s).strip('_')
            return s2 or "filter"
    if digits:
        return f"filter_{int(digits[0])}"
    s2 = re.sub(r'[^a-z0-9]+', '_', s).strip('_')
    return s2 or "unnamed"

# CSV
def read_filter_frobenius_map(path: str) -> Dict[str, float]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise IOError(f"Nie można wczytać {path}: {e}")

    cols_low = {c.strip().lower(): c for c in df.columns}
    if 'filter' in cols_low:
        fcol = cols_low['filter']
    else:
        fcol = df.columns[0]
    if 'frobenius' in cols_low:
        frob_col = cols_low['frobenius']
    else:
        frob_col = df.columns[1] if len(df.columns) > 1 else None
    if frob_col is None:
        raise ValueError(f"Brak kolumny Frobeniusa w pliku {path}")

    fmap: Dict[str, float] = {}
    for _, row in df[[fcol, frob_col]].dropna(subset=[fcol]).iterrows():
        orig = str(row[fcol]).strip()
        try:
            val = float(row[frob_col])
        except Exception:
            try:
                val = float(str(row[frob_col]).replace(',', '.'))
            except Exception:
                val = np.nan
        key = normalize_filter_name(orig)
        fmap[key] = val
    return fmap

def extract_pct_idx_from_filename(fname: str) -> Tuple[int,int]:
    m = RE_MODEL.search(fname)
    if m:
        try:
            return int(m.group(1)), int(m.group(2))
        except:
            return -1, -1
    m2 = RE_MODEL_FALLBACK.search(fname)
    if m2:
        try:
            return int(m2.group(1)), int(m2.group(2))
        except:
            return -1, -1
    return -1, -1

def compare_and_aggregate_newfrobenius_only(directories: List[str], group_indices: List[List[int]],
                                            reference_path: str, new_files: List[str]) -> pd.DataFrame:
    try:
        ref_map = read_filter_frobenius_map(reference_path)
    except Exception as e:
        raise RuntimeError(f"Nie udało się wczytać pliku referencyjnego {reference_path}: {e}")
    ref_set = set(ref_map.keys())
    print(f"[INFO] Referencyjnych filtrów (znormalizowanych): {len(ref_set)}")

    idx_to_group = {}
    for gi, idx_list in enumerate(group_indices, start=1):
        for idx in idx_list:
            idx_to_group[idx] = gi

    serial_counter = defaultdict(int)
    rows = []

    for dir_idx, directory in enumerate(directories):
        if not os.path.isdir(directory):
            print(f"[WARN] Brak katalogu: {directory} -> pomijam")
            continue
        group_num = idx_to_group.get(dir_idx, None)
        if group_num is None:
            continue

        cohort_name = os.path.basename(os.path.normpath(directory))
        if cohort_name.lower() in ('stats','metrics','high','data'):
            cohort_name = os.path.basename(os.path.dirname(os.path.normpath(directory)))

        fnames = sorted(os.listdir(directory))
        for fn in fnames:
            if not (fn.startswith("best") and fn.endswith("filters_above_tresh.csv")):
                continue
            fp = os.path.join(directory, fn)
            try:
                comp_map = read_filter_frobenius_map(fp)
            except Exception as e:
                print(f"[WARN] Nie można wczytać {fp}: {e} -> pomijam")
                continue

            pct, idx = extract_pct_idx_from_filename(fn)
            pct_label = pct if pct >= 0 else -1
            key_for_serial = (group_num, pct_label)
            serial_counter[key_for_serial] += 1
            serial = serial_counter[key_for_serial]

            pct_short = str(int(pct)) if pct >= 0 else "NA"
            model_key = f"{group_num}_{pct_short}_{serial}"

            comp_set = set(comp_map.keys())
            matching_set = comp_set & ref_set
            new_set = comp_set - ref_set
            lost_set = ref_set - comp_set

            matching_total = len(matching_set)
            new_total = len(new_set)
            lost_total = len(lost_set)

            new_gt_low = sum(1 for f in new_set if np.isfinite(comp_map.get(f, np.nan)) and comp_map.get(f, np.nan) > TH_LOW)
            new_gt_high = sum(1 for f in new_set if np.isfinite(comp_map.get(f, np.nan)) and comp_map.get(f, np.nan) > TH_HIGH)

            rows.append({
                'model_key': model_key,
                'group': group_num,
                'pct': pct,
                'serial': serial,
                'cohort': cohort_name,
                'matching_total': int(matching_total),
                'new_total': int(new_total),
                'new_gt0.04': int(new_gt_low),
                'new_gt0.5': int(new_gt_high),
                'lost_total': int(lost_total),
                'count_files': 1
            })

    # Process new_files individually (ensure none are skipped)
    ext_serial_counter = defaultdict(int)
    for nf in new_files:
        if not os.path.isfile(nf):
            print(f"[WARN] Skrajny plik nie istnieje: {nf} -> pomijam")
            continue
        try:
            comp_map = read_filter_frobenius_map(nf)
        except Exception as e:
            print(f"[WARN] Nie można wczytać skrajnego pliku {nf}: {e} -> pomijam")
            continue
        fname = os.path.basename(nf)
        pct, idx = extract_pct_idx_from_filename(fname)
        pct_short = str(int(pct)) if pct >= 0 else "NA"
        key_for_serial = ('EXT', pct_short)
        ext_serial_counter[key_for_serial] += 1
        serial = ext_serial_counter[key_for_serial]

        model_key = f"0_{pct_short}_{serial}"  # group 0 reserved for external/reference
        comp_set = set(comp_map.keys())
        matching_set = comp_set & ref_set
        new_set = comp_set - ref_set
        lost_set = ref_set - comp_set

        matching_total = len(matching_set)
        new_total = len(new_set)
        lost_total = len(lost_set)
        new_gt_low = sum(1 for f in new_set if np.isfinite(comp_map.get(f, np.nan)) and comp_map.get(f, np.nan) > TH_LOW)
        new_gt_high = sum(1 for f in new_set if np.isfinite(comp_map.get(f, np.nan)) and comp_map.get(f, np.nan) > TH_HIGH)

        rows.append({
            'model_key': model_key,
            'group': 0,
            'pct': pct,
            'serial': serial,
            'cohort': None,
            'matching_total': int(matching_total),
            'new_total': int(new_total),
            'new_gt0.04': int(new_gt_low),
            'new_gt0.5': int(new_gt_high),
            'lost_total': int(lost_total),
            'count_files': 1
        })
        print(f"[INFO] Przetworzono skrajny plik: {nf} -> model_key {model_key}")

    df = pd.DataFrame(rows)
    df = df.sort_values(by=['group', 'pct', 'serial'], na_position='last').reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Zapisano CSV: {out_csv} (wierszy: {len(df)})")
    return df

# Plotting
def short_label(model_key: str) -> str:
    parts = model_key.split('_')
    if len(parts) >= 3:
        grp = parts[0]
        pct = parts[1]
        serial = parts[2]
        try:
            pct_i = int(pct)
            pct_str = str(pct_i)
        except Exception:
            pct_str = pct
        return f"{grp}_{pct_str}_{serial}"
    return model_key

def plot_models_pages_newonly(df, out_png, page_size=100, max_models=202, max_xticks=100):
    df_use = df.copy().reset_index(drop=True).iloc[:max_models]
    n_models = len(df_use)
    if n_models == 0:
        raise ValueError("Brak danych.")
    pages = math.ceil(n_models / page_size)
    fig_width = 20
    fig_height_per = 5
    fig, axes = plt.subplots(pages, 1, figsize=(fig_width, fig_height_per * pages), squeeze=False)
    axes = axes.flatten()

    color_matching = '#2ca02c'
    color_lost = '#d62728'
    new_shades = ['#bf5700', '#ff8a50', '#ffd6bf']

    for p in range(pages):
        ax = axes[p]
        start = p*page_size
        end = min(start + page_size, n_models)
        sub = df_use.iloc[start:end].reset_index(drop=True)
        n = len(sub)
        x = np.arange(n)
        width = 0.25

        matching = sub['matching_total'].astype(int).values
        lost = sub['lost_total'].astype(int).values
        new_total = sub['new_total'].astype(int).values
        new_low = (new_total - sub['new_gt0.04'].astype(int).values)
        new_mid = (sub['new_gt0.04'].astype(int).values - sub['new_gt0.5'].astype(int).values)
        new_high = sub['new_gt0.5'].astype(int).values

        ax.bar(x - width, matching, width, color=color_matching, edgecolor='black')
        ax.bar(x, lost, width, color=color_lost, edgecolor='black')
        ax.bar(x + width, new_low, width, color=new_shades[2], edgecolor='black')
        ax.bar(x + width, new_mid, width, bottom=new_low, color=new_shades[1], edgecolor='black')
        ax.bar(x + width, new_high, width, bottom=(new_low + new_mid), color=new_shades[0], edgecolor='black')

        short_labels = [short_label(k) for k in sub['model_key'].astype(str).tolist()]
        if n <= max_xticks:
            ticks = x
            labels = short_labels
        else:
            step = max(1, n // max_xticks)
            ticks = x[::step]
            labels = [short_labels[i] for i in ticks]

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_xlim(-1, max(1, n))
        ymax = max(1, np.nanmax(np.vstack([matching, lost, new_total])))
        ax.set_ylim(0, ymax * 1.15)
        ax.grid(axis='y', linestyle='--', alpha=0.25)

        for i_idx in range(n):
            if (i_idx % max(1, math.ceil(n/10))) == 0:
                pass

        if p == 2:
            import matplotlib.patches as mpatches
            legend = [
                mpatches.Patch(facecolor=color_matching, edgecolor='black', label='pasujące względem modelu alt_last'),
                mpatches.Patch(facecolor=color_lost, edgecolor='black', label='stracone względem modelu alt_last'),
                mpatches.Patch(facecolor=new_shades[0], edgecolor='black', label='nowe (> 0.5)'),
                mpatches.Patch(facecolor=new_shades[1], edgecolor='black', label='nowe (0.04-0.5]'),
                mpatches.Patch(facecolor=new_shades[2], edgecolor='black', label='nowe (<= 0.04)')
            ]
            ax.legend(handles=legend, loc='lower right', fontsize=12)


    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.suptitle("Klasyfikacja filtrów po dotrenowaniu w zalezności od normy Frobeniusa", fontsize=14, y=0.995, weight='bold')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"[INFO] Zapisano wykres: {out_png}")
    plt.show()
    plt.close(fig)

# Main
if __name__ == "__main__":
    # df = compare_and_aggregate_newfrobenius_only(directories, group_indices, reference_file, new_files)
    df = pd.read_csv(folder_path + 'model_filters_summary_newonly_frobenius.csv')
    plot_models_pages_newonly(df, out_png, page_size=page_size, max_models=max_models, max_xticks=max_xticks)
