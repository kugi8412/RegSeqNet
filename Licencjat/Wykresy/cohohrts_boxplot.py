#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

"""
Parsowanie listy plików CSV z kolumnami Filter,Frobenius,
obliczanie średniej normy Frobeniusa dla każdego filtra (filter_N)
i rysowanie wykresu słupkowego posortowanego malejąco.

Użycie:
    file_list = ["path/to/file1.csv", "path/to/file2.csv", ...]
    uruchom skrypt -> zapisuje: filter_frobenius_means.csv i filter_frobenius_means_desc.png
"""

base_dir = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/data/test_output/cohorts"
# best_high_080_5_filters_below_tresh.csv
cohorts = [base_dir + f"/{c}/stats/best_high_{p}_{i}_filters_above_tresh.csv" for p in ['040','060','080','020'] for i in range(1,6) for c in ['c1','c2', 'c3','c4','c5','c6','c7','c8','c9','c10']]
# cohorts = [base_dir + f"/{c}/stats/best_high_{p}_{i}_filters_above_tresh.csv" for p in ['040','060','080','020'] for i in range(1,6) for c in ['c8','c9','c10']]

file_list = cohorts

out_csv = "filter_frobenius_summary.csv"     # zapis z count,sum,mean
out_png = "filter_frobenius_3panels_mean.png"
show_values_over_bars = False  # True -> wypisuje wartości nad słupkami (może być nieczytelne przy 100 słupkach)

# ---------------- FUNKCJE ----------------
def safe_float(x):
    """Konwertuje wartość do float, obsługuje przecinek jako separator dziesiętny."""
    try:
        return float(x)
    except Exception:
        try:
            return float(str(x).replace(',', '.'))
        except Exception:
            return None

def collect_sum_count_from_files(file_list):
    """
    Przetwarza listę plików .csv i zwraca słownik:
      sums[filter_label] = suma wartości Frobenius z wszystkich plików
      counts[filter_label] = liczba wystąpień (składników)
    Oczekuje kolumn 'Filter' oraz 'Frobenius' (heurystycznie dopasowuje).
    """
    filter_rx = re.compile(r"filter[_\-]?(\d+)", flags=re.IGNORECASE)
    sums = defaultdict(float)
    counts = defaultdict(int)

    for fp in file_list:
        if not os.path.isfile(fp):
            print(f"[WARN] Plik nie istnieje: {fp} -> pomijam")
            continue
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[WARN] Nie można wczytać {fp}: {e} -> pomijam")
            continue

        # znajdź kolumny: tolerancyjnie
        cols_lower = {c.strip().lower(): c for c in df.columns}
        filter_col = cols_lower.get('filter')
        frob_col = cols_lower.get('frobenius')
        if filter_col is None or frob_col is None:
            for c in df.columns:
                lc = c.strip().lower()
                if filter_col is None and ('filter' in lc or 'filt' in lc):
                    filter_col = c
                if frob_col is None and ('frob' in lc or 'frobenius' in lc or 'norm' in lc):
                    frob_col = c

        if filter_col is None or frob_col is None:
            print(f"[WARN] {fp}: brak kolumn 'Filter'/'Frobenius' -> pomijam")
            continue

        # iteruj wiersze
        for _, row in df[[filter_col, frob_col]].dropna().iterrows():
            raw_name = str(row[filter_col]).strip()
            m = filter_rx.search(raw_name)
            if m:
                label = f"filter_{int(m.group(1))}"
            else:
                # fallback: uproszczona, zastąp spacje podkreślnikiem
                label = re.sub(r'\s+', '_', raw_name)

            val = safe_float(row[frob_col])
            if val is None:
                continue
            sums[label] += val
            counts[label] += 1

    return sums, counts

def build_summary_dataframe(sums, counts, top_n=None, sort_desc=True):
    """
    Buduje DataFrame z kolumnami: filter_label, count, sum_frobenius, mean_frobenius.
    Sortuje wg mean (domyślnie malejąco). Można ograniczyć do top_n rekordów.
    """
    labels = sorted(set(list(sums.keys()) + list(counts.keys())))
    rows = []
    for lbl in labels:
        c = counts.get(lbl, 0)
        s = sums.get(lbl, 0.0)
        mean = (s / c) if c > 0 else np.nan
        rows.append((lbl, c, s, mean))
    df = pd.DataFrame(rows, columns=['filter_label', 'count', 'sum_frobenius', 'mean_frobenius'])
    df = df.sort_values(by='mean_frobenius', ascending=not sort_desc).reset_index(drop=True)
    if top_n is not None:
        df = df.iloc[:top_n].copy()
    return df

def plot_three_vertical_panels_from_df(df, out_png, per_panel=100, show_vals=False):
    """
    df: DataFrame posortowany wg mean_frobenius malejąco. Rysuje do 3 paneli pionowo, każdy per_panel filtrów.
    """
    total = len(df)
    if total == 0:
        print("[INFO] Brak danych do rysowania.")
        return

    # ograniczenie do max 300
    if total > 300:
        df = df.iloc[:300].copy()
        total = 300

    # liczba paneli (maks 3)
    from math import ceil
    n_panels = 4
    # rozdzielenie na n_panels równych kawałków (równomiernie)
    base = total // n_panels
    rem = total % n_panels
    splits = []
    start = 0
    for i in range(n_panels):
        size = base + (1 if i < rem else 0)
        splits.append((start, start + size))
        start += size

    splits = [(0, 7), (7, 58 - 10), (58 - 10, 99), (99, 300)]

    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 4*n_panels))
    if n_panels == 1:
        axes = [axes]

    for ax_idx, (s, e) in enumerate(splits):
        chunk = df.iloc[s:e]
        labels = chunk['filter_label'].tolist()
        values = chunk['mean_frobenius'].values
        x = np.arange(len(labels))

        bars = axes[ax_idx].bar(x, values, edgecolor='black')

        # ustawienie xticks co k, by nie zapchać etykiet
        n_labels = len(labels)
        if n_labels <= 20:
            step = 1
        elif n_labels <= 50:
            step = 2
        elif n_labels <= 100:
            step = 5
        else:
            step = max(1, n_labels // 20)
        
        step = 1 # change
        # 269 wszystkie, a 254 istotne (60)
        tick_positions = np.arange(0, n_labels, step)
        tick_labels = [labels[i][7:] for i in tick_positions]
        axes[ax_idx].set_xticks(tick_positions)
        if ax_idx == n_panels - 1:
            axes[ax_idx].set_xticklabels(tick_labels, rotation=75, fontsize=6)
        else:
            axes[ax_idx].set_xticklabels(tick_labels, rotation=75, fontsize=8)
        print(tick_labels)

        axes[ax_idx].set_ylabel("Średnia norma")
        # axes[ax_idx].set_title(f"Filters {s+1}-{e} (of {total})")

        if show_vals:
            vmax = np.nanmax(values) if len(values)>0 else 0
            for rect, val in zip(bars, values):
                height = rect.get_height()
                axes[ax_idx].text(rect.get_x() + rect.get_width()/2., height + 0.01*vmax,
                                  f"{val:.3f}", ha='center', va='bottom', fontsize=5, rotation=90)
        axes[ax_idx].grid(axis='y', linestyle='--', alpha=0.4)
    
    P = []
    for l in labels:
        P.append(l[7:])
    
    print(P)

    # print(axes)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(hspace=0.4, bottom=0.05)
    fig.suptitle("Posortowana malejąco średnia norma Frobeniusa istotnych filtrów", fontsize=14, y=0.995, weight='bold')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {out_png}")
    plt.show()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    sums, counts = collect_sum_count_from_files(file_list)
    if len(sums) == 0:
        print("[INFO] Nie zebrano żadnych wartości Frobenius. Sprawdź pliki.")
    else:
        df_summary = build_summary_dataframe(sums, counts, top_n=None, sort_desc=True)
        # zapisz CSV
        df_summary.to_csv(out_csv, index=False)
        print(f"Zapisano summary CSV: {out_csv} (rekordów: {len(df_summary)})")
        # rysuj panele (maks 300 filtrów -> 3 panele x 100)
        plot_three_vertical_panels_from_df(df_summary, out_png=out_png, per_panel=100, show_vals=show_values_over_bars)