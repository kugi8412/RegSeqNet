import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# === PARAMETRY ===
file1 = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/filters/alt_last/alt1_last_filter.txt"
file2 = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/filters/highsignal_last/high_signal4_last_filter.txt"
threshold = 0.05

# === FUNKCJE ===
def read_filters(filepath: str) -> np.ndarray:
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

def frobenius_norms(filters):
    return np.linalg.norm(filters, axis=1)

def plot_norm_distribution(norms1, norms2, threshold):
    below1 = np.sum(norms1 < threshold)
    above1 = np.sum(norms1 >= threshold)
    below2 = np.sum(norms2 < threshold)
    above2 = np.sum(norms2 >= threshold)

    labels = ['< threshold', '≥ threshold']
    x = np.arange(len(labels))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.35
    ax.bar(x - bar_width/2, [below1, above1], width=bar_width, label='Model alt', color='cyan')
    ax.bar(x + bar_width/2, [below2, above2], width=bar_width, label='Model high', color='salmon')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Liczba filtrów")
    ax.set_title("Rozkład liczby filtrów względem progu normy Frobeniusa")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_sorted_norms(norms1, norms2, threshold):
    filtered1 = np.sort(norms1[norms1 >= threshold])
    filtered2 = np.sort(norms2[norms2 >= threshold])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(filtered1, label='Model alt', color='blue')
    ax.plot(filtered2, label='Model high', color='red')
    ax.set_title(f"Normy Frobeniusa (≥ {threshold:.1e}), posortowane")
    ax.set_ylabel("Norma Frobeniusa")
    ax.set_xlabel("Indeks (po sortowaniu)")
    ax.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_pca(filters1, filters2, norms1, norms2, threshold):
    idx1 = norms1 >= threshold
    idx2 = norms2 >= threshold
    filt1 = filters1[idx1]
    filt2 = filters2[idx2]
    norms1 = norms1[idx1]
    norms2 = norms2[idx2]

    pca1 = PCA(n_components=2).fit_transform(filt1)
    pca2 = PCA(n_components=2).fit_transform(filt2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, pca, norms, title in zip(axes, [pca1, pca2], [norms1, norms2], ['Model 1', 'Model 2']):
        sc = ax.scatter(pca[:, 0], pca[:, 1], c=norms, cmap='viridis', edgecolor='k', alpha=0.9)
        ax.set_title(f'PCA filtrów ({title})')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        plt.colorbar(sc, ax=ax, label='Norma Frobeniusa')

    plt.suptitle("Redukcja PCA filtrów (z normą > threshold)", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()


MY_FILTERS = [
    205, 105, 267, 153,  30, 229,  76,  27, 104, 113,
    167, 182, 191,  11,  71, 141,  53,  65, 285, 247,
    188, 103, 292, 216, 138, 288,  50, 193,  48, 299,
    148, 158,  17, 203,  18, 225, 121,  23, 221, 200,
    108,  70, 187, 214, 236,  62,  64,  38, 140
]

def load_filters(filepath: str) -> np.ndarray:
    filters = []
    with open(filepath, 'r') as f:
        current_filter = []
        for line in f:
            line = line.strip()
            if line.startswith('>filter_'):
                if current_filter:
                    filters.append(np.array(current_filter, dtype=float))
                current_filter = []
            elif line:
                current_filter.append([float(x) for x in line.split()])
        if current_filter:
            filters.append(np.array(current_filter, dtype=float))
    return np.stack(filters) if filters else np.empty((0, 4, 19))

def compute_metrics(filters: np.ndarray, selected: List[int]):
    valid_indices = [idx for idx in selected if idx < filters.shape[0]]
    if not valid_indices:
        return {}, {}
    
    vecs = filters.reshape(filters.shape[0], -1)
    norms = np.linalg.norm(vecs, axis=1)
    units = vecs / (norms[:, np.newaxis] + 1e-12)
    
    diff_metrics = {}
    ang_metrics = {}
    
    for idx in valid_indices:
        others = [j for j in valid_indices if j != idx and norms[j] > 0.005]
        
        # Oblicz różnicę norm
        if others and norms[idx] > 0.005:
            diff = np.min(np.abs(norms[others] - norms[idx]))
        else:
            diff = 0.0
        # Oblicz kąt
        if others and norms[idx] > 0.05:
            cos_sim = np.dot(vecs[others], vecs[idx]) / (norms[others] * norms[idx])
            min_ang = min(np.ones(cos_sim.shape) - cos_sim)
        else:
            min_ang = 0.0

        diff_metrics[idx] = diff
        ang_metrics[idx] = min_ang
    
    return diff_metrics, ang_metrics

def rank_and_aggregate(filepaths: List[str], selected: List[int], gamma: float = 0.5):
    all_diff_ranks = {i: [] for i in selected}
    all_ang_ranks = {i: [] for i in selected}
    
    for fp in filepaths:
        filters = load_filters(fp)
        if filters.size == 0:
            continue
            
        diff_metrics, ang_metrics = compute_metrics(filters, selected)
        
        # Konwersja do rank
        sorted_diff = sorted(diff_metrics.items(), key=lambda x: -x[1])
        diff_ranks = {k: i+1 for i, (k, v) in enumerate(sorted_diff)}
        
        sorted_ang = sorted(ang_metrics.items(), key=lambda x: -x[1])
        ang_ranks = {k: i+1 for i, (k, v) in enumerate(sorted_ang)}
        
        for idx in diff_metrics:
            all_diff_ranks[idx].append(diff_ranks.get(idx, 0))
            all_ang_ranks[idx].append(ang_ranks.get(idx, 0))
    
    # Oblicz średnie rangi
    avg_scores = {}
    for idx in selected:
        avg_diff = np.mean(all_diff_ranks[idx]) if all_diff_ranks[idx] else 0
        avg_ang = np.mean(all_ang_ranks[idx]) if all_ang_ranks[idx] else 0
        avg_scores[idx] = gamma * avg_diff + (1 - gamma) * avg_ang
    
    print("Average Scores:", avg_scores)
    return avg_scores

def plot_results(scores: Dict[int, float], output_path: str = 'filter_ranking.png'):
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    indices, values = zip(*sorted_scores)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(indices)), values, color='c', alpha=0.75)
    plt.xticks(range(len(indices)), indices, rotation=90)
    plt.xlabel('Indeksy filtrów', fontsize=12)
    plt.ylabel('Uśredniony ranking', fontsize=12)
    plt.title('Średni ranking na podstawie minimalnej ', fontsize=16, weight='bold')
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    files = []
    path = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/data/test_output/cohorts/"
    for c in ['c8', 'c9', 'c10']:
        files += [os.path.join(path, c + '/filter', f'best_high_{p}_{i}_filter.txt') for p in ['040','060','080','020'] for i in range(1,6)]
    final_scores = rank_and_aggregate(files, MY_FILTERS, gamma=1.00)
    plot_results(final_scores, 'final_filter_ranking.png')

    filters1 = read_filters(file1)
    filters2 = read_filters(file2)
    norms1 = frobenius_norms(filters1)
    norms2 = frobenius_norms(filters2)

    plot_norm_distribution(norms1, norms2, threshold)
    plot_sorted_norms(norms1, norms2, threshold)
    plot_pca(filters1, filters2, norms1, norms2, threshold)
