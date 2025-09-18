import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Tuple

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

def plot_filter_counts(norms1: np.ndarray, norms2: np.ndarray, threshold: float):
    count1 = np.sum(norms1 >= threshold)
    count2 = np.sum(norms2 >= threshold)
    labels = ['Model 1', 'Model 2']
    vals   = [count1, count2]

    fig, ax = plt.subplots(figsize=(12,6))
    bars = ax.bar(labels, vals, color=['C0','C1'], alpha=0.8)
    ax.set_ylabel("Liczba filtrów norm ≥ próg")
    ax.set_title(f"Liczba filtrów powyżej progu {threshold:.1e}")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.5, str(v),
                ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

def plot_pca(filters: np.ndarray, norms: np.ndarray, threshold: float, title: str):
    mask = norms >= threshold
    vecs = filters.reshape(filters.shape[0], -1)
    selected = vecs[mask]
    selected_norms = norms[mask]
    if selected.shape[0] < 2:
        print(f"Za mało filtrów ≥ {threshold:.1e} w {title} do PCA.")
        return

    pca = PCA(n_components=2)
    coords = pca.fit_transform(selected)
    var_pct = pca.explained_variance_ratio_.sum() * 100

    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.scatter(coords[:,0], coords[:,1],
                    c=selected_norms, cmap='viridis', edgecolor='k')
    ax.set_title(f"{title}\nPCA 2D - {var_pct:.1f}% wariancji")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(sc, ax=ax, label="Norma Frobeniusa")
    plt.tight_layout()
    plt.show()

def main(file1: str, file2: str, threshold: float):
    # Wczytanie i obliczenia
    F1 = read_filters(file1)
    F2 = read_filters(file2)
    norms1 = compute_frobenius_norms(F1)
    norms2 = compute_frobenius_norms(F2)

    # 1) Histogramy liczb filtrów powyżej progu
    plot_filter_counts(norms1, norms2, threshold)

    # 2) Sortowane normy powyżej progu
    s1 = np.sort(norms1[norms1>=threshold])
    s2 = np.sort(norms2[norms2>=threshold])
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(s1, label='Model Alt')
    ax.plot(s2, label='Model High')
    ax.set_title(f"Normy >= {threshold:.1e}, posortowane")
    ax.set_xlabel("Porządek filtra")
    ax.set_ylabel("Norma Frobeniusa")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3) PCA dla obu
    plot_pca(F1, norms1, threshold, "Model Alt")
    plot_pca(F2, norms2, threshold, "Model High")

if __name__ == '__main__':
    # Podmień ścieżki na swoje pliki:
    file1 = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/filters/alt_last/alt1_last_filter.txt"
    file2 = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/filters/highsignal_last/high_signal4_last_filter.txt"
    threshold = 0.05
    main(file1, file2, threshold)
