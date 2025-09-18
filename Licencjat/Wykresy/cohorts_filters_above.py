import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

def load_filters_from_dir(filter_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Wczytuje pliki origin_* i best_* w katalogu i zwraca
    słownik model_key -> {'origin':…, 'best':…}.
    """
    models = {}
    for fname in sorted(os.listdir(filter_dir)):
        if not fname.endswith('.txt'): 
            continue
        if fname.startswith('origin_') or fname.startswith('best_'):
            prefix, rest = fname.split('_', 1)
            model_key = rest[:-4] + '_' + os.path.basename(filter_dir)
            path = filter_dir + '\\' + fname
            filters, current = [], []
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('>filter_'):
                        if current:
                            filters.append(np.array(current))
                        current = []
                    elif line:
                        current.append(list(map(float, line.split())))
                if current:
                    filters.append(np.array(current))
            models.setdefault(model_key, {})[prefix] = np.stack(filters)
    return models

def compute_l2_norms(filters: np.ndarray) -> np.ndarray:
    """Funckaj oblicza normę Frobeniusa dla każdej macierzy (num_filters,4,19) → (num_filters,)."""
    return np.linalg.norm(filters.reshape(filters.shape[0], -1), axis=1)

def plot_average_norms_filtered(
    models: Dict[str, Dict[str, np.ndarray]],
    thresholds: List[float],
    top_range: Tuple[int,int],
    special_filters: List[int],
    output_dir: str = 'outputs\\average_filtered'
):
    os.makedirs(output_dir, exist_ok=True)

    # To CHANGE (change, best)
    all_origin = [compute_l2_norms(m['best']) 
                  for m in models.values() if 'best' in m]
    num_filters = all_origin[0].shape[0]

    for thr in thresholds:
        # sumy i zliczenia dla każdego filtra
        sums = np.zeros(num_filters)
        counts = np.zeros(num_filters)
        for norms in all_origin:
            mask = norms > thr
            sums[mask] += norms[mask]
            counts[mask] += 1

        # średnia tam, gdzie count>0
        mean_norm = np.full(num_filters, np.nan)
        valid = counts > 0
        mean_norm[valid] = sums[valid] / counts[valid]

        # sort malejąco, wybierz top_range
        sorted_idxs = np.argsort(np.nan_to_num(mean_norm, nan=-np.inf))[::-1]
        start, end = top_range
        sel = [i for i in sorted_idxs if valid[i]][start:end]
        vals = mean_norm[sel]

        # rysuj
        fig, ax = plt.subplots(figsize=(12, 4))
        x = np.arange(len(sel))
        colors = ['red' if idx in special_filters else 'blue' for idx in sel]
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([str(idx) for idx in sel], rotation=90)
        ax.set_xlabel('Filter')
        ax.set_ylabel('Average L2 Norm')
        ax.set_title(f'Aggregated Filters {start+1}-{end}, origin L2 > {thr}')
        ax.plot([], [], color='red', label='Special filters')
        ax.plot([], [], color='blue', label='Other filters')
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{output_dir}\\aggregate_{start+1}_{end}_thr_{thr}.png")
        plt.close(fig)

def plot_l2_diff_all_models(
    models: Dict[str, Dict[str, np.ndarray]],
    thresholds: List[float],
    output_dir: str = 'outputs\\l2_differences'
):
    os.makedirs(output_dir, exist_ok=True)
    # rozszerzamy o +inf
    bounds = thresholds + [np.inf]

    for i in range(len(thresholds)):
        lo, hi = thresholds[i], bounds[i+1]
        diffs = []
        for data in models.values():
            if 'origin' not in data or 'best' not in data: 
                continue
            l2_o = compute_l2_norms(data['origin'])
            l2_b = compute_l2_norms(data['best'])
            mask = (l2_o > lo) & (l2_o <= hi)
            diffs.extend(np.abs(l2_b[mask] - l2_o[mask]))

        if not diffs:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(diffs, bins=30, color='gray', edgecolor='black')
        ax.set_title(f'All models |ΔL2| for origin ∈({lo}, {hi}]')
        ax.set_xlabel('|L2_best - L2_origin|')
        ax.set_ylabel('Count')
        fig.tight_layout()
        fig.savefig(f"{output_dir}\\diff_all_{lo:.3f}_{hi:.3f}.png")
        plt.close(fig)

def main(
    base_dir: str,
    cohorts: List[str],
    thresholds_avg: List[float],
    top_range: Tuple[int,int],
    special_filters: List[int]
):
    models = {}
    for cohort in cohorts:
        dir_f = base_dir + "\\" + cohort + "\\filter"
        if os.path.isdir(dir_f):
            sub = load_filters_from_dir(dir_f)
            models.update(sub)

    plot_average_norms_filtered(models, thresholds_avg, top_range, special_filters)
    plot_l2_diff_all_models(models, thresholds_avg)

if __name__ == '__main__':
    folder_path = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/"
    base_dir = folder_path + "/data/test_output/cohorts/"
    selected_cohorts = ['c2', 'c3', 'c5', 'c6', 'c7']
    thresholds_avg = [0.01, 0.6, 0.8, 1.0]  # progi do agregacji
    top_range = (0, 50)  # filtry 50-100
    special_filters = np.array(['11', '18', '23', '27', '30',
                        '48', '50', '53', '65', '70',
                        '71', '76', '103', '104', '105',
                        '108', '113', '121', '136', '138',
                        '141', '148', '153', '158', '159',
                        '167', '182', '188', '191', '193',
                        '200', '203', '205', '214', '216',
                        '221', '225', '229', '247', '267',
                        '285', '288', '292', '299',
                        '17', '236'], dtype='int64')  # przykładowo
    main(base_dir, selected_cohorts, thresholds_avg, top_range, special_filters)
    print("Done")
