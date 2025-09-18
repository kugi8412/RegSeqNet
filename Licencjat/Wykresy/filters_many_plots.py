#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List

def plot_models_vertical_pages(
    df: pd.DataFrame,
    out_png: str = "models_vertical_pages.png",
    max_models: int = 200,
    page_size: int = 100,
    max_xticks: int = 20,
    figsize_per_page: Tuple[int,int] = (18, 6),
    colors: Tuple[str,str,str] = ('#2ca02c','#d62728','#ff7f0e')
) -> List[str]:
    """
    Rysuje modele w pionowych 'stronach' (subplots ułożonych jeden pod drugim).
    - df: DataFrame zawierający kolumny: model_key, group, pct, idx, matching, lost, non_matching, count_files
    - max_models: maksymalna liczba modeli do wzięcia (domyślnie 200)
    - page_size: ile modeli przypada na jeden subplot (domyślnie 100)
    - max_xticks: ile etykiet X na subplot (reszta pomijana dla czytelności)
    - figsize_per_page: rozmiar pojedynczego subplotu (szerokość, wysokość) — finalne figsize = (szer, height*pages)
    - colors: kolory (matching, lost, new)
    Zwraca listę użytych model_key (kolejność).
    """

    required_cols = {'model_key','matching','lost','non_matching','count_files','group'}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"DataFrame musi zawierać kolumny: {required_cols}")

    df_use = df.copy().reset_index(drop=True).iloc[:max_models].reset_index(drop=True)
    n_models = len(df_use)
    if n_models == 0:
        raise ValueError("Brak modeli w DataFrame.")

    pages = math.ceil(n_models / page_size)

    fig_width = figsize_per_page[0]
    fig_height = figsize_per_page[1] * pages
    fig, axes = plt.subplots(pages, 1, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()

    used_keys = []

    for page_idx in range(pages):
        ax = axes[page_idx]
        start = page_idx * page_size
        end = min(start + page_size, n_models)
        sub = df_use.iloc[start:end].reset_index(drop=True)
        used_keys.extend(sub['model_key'].astype(str).tolist())

        n = len(sub)
        x = np.arange(n)
        width = 0.25

        # ensure for float
        matching = sub['matching'].astype(float).fillna(0).values
        lost = sub['lost'].astype(float).fillna(0).values
        new = sub['non_matching'].astype(float).fillna(0).values

        ax.bar(x - width, matching, width, label='matching', color=colors[0])
        ax.bar(x, lost, width, label='lost', color=colors[1])
        ax.bar(x + width, new, width, label='new', color=colors[2])

        if n <= max_xticks:
            tick_pos = x
            tick_labels = sub['model_key'].astype(str).tolist()
        else:
            step = max(1, n // max_xticks)
            tick_pos = x[::step]
            tick_labels = [str(k) for k in sub['model_key'].iloc[::step].tolist()]

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
        ax.set_xlim(-1, max(1, n))
        ymax = max(1.0, np.nanmax(np.vstack([matching, lost, new])))
        ax.set_ylim(0, ymax * 1.12)

        for i_idx in range(n):
            if (i_idx % max(1, math.ceil(n/max(10, max_xticks)))) == 0:
                ax.text(i_idx, -0.06*ymax, f"n={int(sub['count_files'].iat[i_idx])}", ha='center', va='top', fontsize=7, rotation=90)

        ax.set_ylabel("Number of filters")
        ax.set_title(f"Models {start+1}–{end} (grouped by cohort sets)")

        # Grid
        ax.grid(axis='y', linestyle='--', alpha=0.25)

        if page_idx == 0:
            ax.legend(loc='upper right')

    for j in range(pages, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.suptitle(f"Aggregated models (total shown: {n_models}) — pages: {pages}", fontsize=14, weight='bold', y=0.995)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return used_keys

# Example using
df_models = pd.read_csv("model_grouped_by_cohort_summary.csv")
used = plot_models_vertical_pages(df_models, out_png="models_200_vertical.png",
                                   max_models=202, page_size=101, max_xticks=200,
                                   figsize_per_page=(20,6))
