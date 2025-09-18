#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu, entropy
from scipy.stats import pearsonr
from scipy.stats import spearmanr, mode
from scipy.stats import kurtosis, skew

"""
# Model
class TestNetwork(nn.Module):
    def __init__(self, seq_len):
        super(TestNetwork, self).__init__()
        pooling_widths = [3, 4, 4]
        num_channels = [30, 20, 20]
        kernel_widths = [5, 3, 3]
        paddings = [int((w - 1) / 2) for w in kernel_widths]

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, num_channels[0], kernel_size=(4, kernel_widths[0]), padding=(0, paddings[0])),
            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[0]), ceil_mode=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(num_channels[0], num_channels[1], kernel_size=(1, kernel_widths[1]), padding=(0, paddings[1])),
            nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[1]), ceil_mode=True))

num_repeats = 100
seq_len = 2000
all_weights = []

for _ in range(num_repeats):
    model = TestNetwork(seq_len)
    weights = model.layer1[0].weight.data.cpu().numpy().flatten()
    all_weights.extend(weights)

all_weights = np.array(all_weights)
plt.figure(figsize=(8, 5))
sns.kdeplot(all_weights, fill=True, color='#d5edea', alpha=1.0, bw_adjust=0.5)
plt.title(f"Rozkład wag pierwszej warstwy konwolucyjnej po inicjalizacji\n", size=24)
plt.xlabel("Wartość wagi", size=18)
plt.ylabel("Gęstość prawdopodobieństwa", size=18)
plt.grid(True)
plt.tight_layout()
plt.show()


def read_filters(filepath: str) -> np.ndarray:
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
                current.append([float(x) for x in line.split()])
        if current:
            filters.append(np.array(current, dtype=float))
    return np.stack(filters, axis=0)

def compute_frobenius_norms(filters: np.ndarray) -> np.ndarray:
    N = filters.shape[0]
    flat = filters.reshape(N, -1)
    return np.linalg.norm(flat, axis=1)

# Ścieżki do plików
file1 = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/filters/alt_last/alt1_last_filter.txt"
F1 = read_filters(file1)
norms1 = compute_frobenius_norms(F1)

# Norm Threshold
threshold = 0.04

# Wyciągamy wszystkie wartości filtrów, których norma > threshold
values1 = F1[norms1 > threshold].ravel()
plt.figure(figsize=(8, 5))
bins = 100

sns.kdeplot(values1, fill=True, color='#d5edea', alpha=1.0, bw_adjust=0.2)
plt.title(f"Rozkład wag istotnych filtrów w modelu alt_last\n", size=24)
plt.xlabel("Wartość wagi", size=16)
plt.ylabel("Gęstość prawdopodobieństwa", size=16)
plt.grid(True)
plt.tight_layout()
plt.show()
"""

def read_filters(filepath: str) -> np.ndarray:
    """ Wczytuje plik z filtrami, zwraca tablicę (N, 4, 19).
    """
    filters = []
    with open(filepath, 'r') as f:
        curr = []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>filter_'):
                if curr:
                    filters.append(np.array(curr, float))
                curr = []
            else:
                curr.append([float(x) for x in line.split()])
        if curr:
            filters.append(np.array(curr, float))
    return np.stack(filters)  # shape (N,4,19)


def fro_norms(filters: np.ndarray) -> np.ndarray:
    """ Flat (N, 4, 19) -> (N, 76) and compute Frobenius norms.
    """
    N = filters.shape[0]
    flat = filters.reshape(N, -1)
    return np.linalg.norm(flat, axis=1)

def plot_two_histograms(
    norms1: np.ndarray,
    norms2: np.ndarray,
    bins: int = 40,
    title_left: str="Model alt_last",
    title_right: str="Model high4_last"
):
    fig, (axL, axR) = plt.subplots(1,2, figsize=(12,4), sharey=True)

    all_norms = np.concatenate([norms1, norms2])
    edges = np.linspace(all_norms.min(), all_norms.max(), bins+1)
    axL.hist(norms1, bins=edges, alpha=0.5, label='Model 1', color='lightblue')
    axL.set_title(title_left)
    axL.set_xlabel("Norma Frobeniusa")
    axL.set_ylabel("Liczba filtrów")
    axL.legend()
    axL.grid(alpha=0.3)

    n1nz = norms1[norms1>0]
    n2nz = norms2[norms2>0]
    edges2 = np.linspace(min(n1nz.min(), n2nz.min()), max(n1nz.max(), n2nz.max()), bins+1)
    axR.hist(n1nz, bins=edges2, alpha=0.5, label='alt_last', color='lightblue')
    axR.hist(n2nz, bins=edges2, alpha=0.5, label='high4_last', color='salmon')
    axR.set_title(title_right)
    axR.set_xlabel("Norma Frobenius")
    axR.legend()
    axR.grid(alpha=0.3)

    fig.suptitle("Porównanie rozkładu norm")
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.show()

def plot_threshold_histograms(
    norms1: np.ndarray,
    norms2: np.ndarray,
    threshold: float,
    bins: int = 30):
    t1 = norms1[norms1 >= threshold]
    t2 = norms2[norms2 >= threshold]

    plot_two_histograms(
        t1, t2, bins=bins,
        title_left=f"Normy ≥ {threshold:.1e}\n(wszystkie)",
        title_right=f"Normy ≥ {threshold:.1e}\n(normy>0)"
    )

def main(file1: str, file2: str, threshold: float):
    F1 = read_filters(file1)
    F2 = read_filters(file2)
    norms1 = fro_norms(F1)
    norms2 = fro_norms(F2)

    plot_two_histograms(norms1, norms2)

    plot_threshold_histograms(norms1, norms2, threshold)


def plot_significant_vs_non(
    sig_count: int,
    non_count: int,
    models: list = ['alt_last', 'high4_last'],
    output_path: str = 'significance_histogram.png'
):
    """
    Rysuje obok siebie porównanie liczby filtrów istotnych i wyzerowanych
    dla dwóch modeli, z legendą poniżej wykresu.
    """
    x = np.arange(len(models))
    sig_vals = [sig_count] * len(models)
    non_vals = [non_count] * len(models)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, sig_vals, width, label='Istotne filtry', color='green')
    bars2 = ax.bar(x + width/2, non_vals, width, label='Wyzerowane filtry', color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Liczba filtrów')
    ax.set_title('Porównanie liczby filtrów istotnych i wyzerowanych', weight='bold')
    ax.set_ylim(0, max(sig_count, non_count) + 30)
    ax.grid(axis='y', alpha=0.5)

    # Adnotations over batrs
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 2,
                str(int(height)), ha='center', va='bottom', fontsize=10)

    # Leggend position
    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.15),
              ncol=2,
              frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def read_filters(path):
    """
    Reads filters from a file.
    Returns a dict: {filter_index: numpy array of shape (num_weights,)}
    """
    filters = {}
    current_idx = None
    values = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>filter_'):
                # save previous
                if current_idx is not None:
                    filters[current_idx] = np.concatenate(values)
                # start new
                current_idx = int(line.split('_')[1])
                values = []
            else:
                # parse line of floats
                row = np.fromstring(line, sep=' ')
                values.append(row)
        # save last
        if current_idx is not None:
            filters[current_idx] = np.concatenate(values)
    return filters

def frob_norm(x):
    """Frobenius norm (Euclidean norm for a vector)."""
    return np.linalg.norm(x)


def compare_base():
    paths = ["C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/filters/alt_last/alt1_last_filter.txt",
              "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/filters/highsignal_last/high_signal4_last_filter.txt",
              "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/filters/first/best000_highsignal_filter.txt"]
    # Read
    filt1 = read_filters(paths[0])
    filt2 = read_filters(paths[1])
    filt3 = read_filters(paths[2])
    norms1 = {idx: frob_norm(arr) for idx, arr in filt1.items()}
    norms2 = {idx: frob_norm(arr) for idx, arr in filt2.items()}
    norms3 = {idx: frob_norm(arr) for idx, arr in filt3.items()}
    selected = [idx for idx, n in norms1.items() if n > 0.5]
    print("Filter indices with Frobenius norm > 0.5 in file1:")
    print(selected)
    print("\nNorms for selected filters:")
    print("Idx\tFile1\tFile2\tFile3")

    for idx in selected:
        n1 = norms1.get(idx, np.nan)
        n2 = norms2.get(idx, np.nan)
        n3 = norms3.get(idx, np.nan)
        print(f"{idx}\t{n1:.4f}\t{n2:.4f}\t{n3:.4f}")

    plt.figure(figsize=(8, 5))
    xs = np.arange(len(selected))
    ys1 = [norms1[idx] for idx in selected]
    ys2 = [norms2[idx] for idx in selected]
    ys3 = [norms3[idx] for idx in selected]
    plt.plot(xs, ys1, marker='o', label='alt_last', color='skyblue', linestyle='', markersize=6, alpha=1.0)
    plt.plot(xs, ys2, marker='s', label='high4_last', color='salmon', linestyle='', markersize=6, alpha=1.0)
    plt.plot(xs, ys3, marker='^', label='alt_high_0', color='lightgreen', linestyle='', markersize=6, alpha=1.0)
    plt.xticks(xs, selected, rotation=75)
    plt.xlabel('Indeksy filtrów')
    plt.ylabel('Norma Frobeniusa')
    plt.title('Normy Frobeniusa filtrów pierwszej warstwy konwolucyjnej', weight='bold')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig('filter_norms_comparison.png', dpi=300)
    plt.show()

    # Average change
    diffs = [np.linalg.norm(filt1[idx] - filt3[idx]) for idx in selected]
    avg_diff = np.mean(diffs)
    print(f"\nAverage Euclidean change (file1 vs file3) over selected filters: {avg_diff:.4f}")

    # Average increase in norms with std
    avg = [norms3[idx] - norms1[idx] for idx in selected if norms3[idx] > 0.5]
    avg_increase = np.mean(avg)
    std_increase = np.std(avg)
    print(f"Average increase in norms (file3 vs file1): {avg_increase:.4f} ± {std_increase:.4f}")

    # 5) KS test on distribution of norms between file2 and file3
    values1 = list(norms1.values())
    values2 = list(norms2.values())
    values3 = list(norms3.values())
    print('MODA:', mode(values1), mode(values2), mode(values3))
    vals1 = [values1[i] for i in range(len(values1)) if values3[i] > 0.5]  # exclude zero norms
    vals3 = [values3[i] for i in range(len(values3)) if values3[i]> 0.5]  # exclude zero norms
    vals2 = [values2[i] for i in range(len(values2)) if values3[i]> 0.5]  # exclude zero norms
    print('MEDIANA:', np.median(vals1), np.median(vals2), np.median(vals3))
    ks_stat, p_value = mannwhitneyu(vals2, vals3)
    pearsonr_stat, pearson_p_value = pearsonr(vals2, vals3)
    spearman_stat, spearman_p_value = spearmanr(vals2, vals3)
    print(f"\nKolmogorov–Smirnov test between file2 and file3 norms:")
    print(f" KS statistic = {ks_stat:.4f}, p-value = {p_value:.4e}")
    print(f" Pearson correlation = {pearsonr_stat:.4f}, p-value = {pearson_p_value:.4e}")
    print(f" Spearman correlation = {spearman_stat:.4f}, p-value = {spearman_p_value:.4e}")
    sns.kdeplot(vals1, label="alt_last", shade=True,color='skyblue')
    sns.kdeplot(vals2, label="high4_last", shade=True,color='salmon')
    sns.kdeplot(vals3, label="alt_high_0", shade=True,color='lightgreen')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False)
    plt.title("Porównanie rozkładów norm Frobeniusa pierwszej warstwy konwolucyjnej", weight='bold')
    plt.xlabel("Norma Frobeniusa")
    plt.ylabel("Gęstość")
    plt.tight_layout()
    plt.show()

    # KL distance
    bins = np.histogram_bin_edges(vals3, bins='auto')
    p1, _ = np.histogram(vals1, bins=bins, density=True)
    p2, _ = np.histogram(vals2, bins=bins, density=True)
    p3, _ = np.histogram(vals3, bins=bins, density=True)

    # Epsilon addition to avoid log(0)
    eps = 1e-12
    p1 += eps; p2 += eps; p3 += eps
    p1 /= p1.sum(); p2 /= p2.sum(); p3 /= p3.sum()

    # KL divergences
    kl_12 = entropy(p1, p2)
    kl_21 = entropy(p2, p1)
    kl_13 = entropy(p1, p3)
    kl_31 = entropy(p3, p1)
    kl_23 = entropy(p2, p3)
    kl_32 = entropy(p3, p2)

    print("\nOdległości KL (Frobenius norms):")
    print(f" D_KL(vals1 || vals2) = {kl_12:.4f}")
    print(f" D_KL(vals2 || vals1) = {kl_21:.4f}")
    print(f" D_KL(vals1 || vals3) = {kl_13:.4f}")
    print(f" D_KL(vals3 || vals1) = {kl_31:.4f}")
    print(f" D_KL(vals2 || vals3) = {kl_23:.4f}")
    print(f" D_KL(vals3 || vals2) = {kl_32:.4f}")

    # Kurtoza
    for name, data in [("alt_last", vals1), ("high4_last", vals2), ("alt_high_0", vals3)]:
        k = kurtosis(data, fisher=True, bias=False)
        s = skew(data, bias=False)
        # Interpretacja kurtozy:
        #  k >  0 -> leptokurtyczne (ostre, ciężkie ogony)
        #  k = 0 -> mezokurtyczne (jak normalne)
        #  k <  0 -> platykurtyczne (płaskie, lekkie ogony)
        print(f"{name}: kurtosis={k:.3f}, skewness={s:.3f}")

if __name__ == '__main__':
    file1 = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/filters/alt_last/alt1_last_filter.txt"
    file2 = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/filters/highsignal_last/high_signal4_last_filter.txt"
    threshold = 0.05
    # main(file1, file2, threshold)
    compare_base()
