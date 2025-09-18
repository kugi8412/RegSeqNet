import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Folder
folder_path = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/"
directories = [folder_path + "data/test_output/cohorts/c2/high/",
               folder_path + "data/test_output/cohorts/c3/high/",
               folder_path + "data/test_output/cohorts/c5/high/",
               folder_path + "data/test_output/cohorts/c6/high/",
               folder_path + "data/test_output/cohorts/c7/high/"]
folder_path += "test_set/best_high/prediction/highsignal/metrics"

# Inicjalizacja przechowywania danych
metrics = {"AUC": [[] for _ in range(4)]}  # Tylko AUC
prefixes = ["020", "040", "060", "080"]
classes = ["PA", "NA", "PI", "NI"]

# Lista katalogów modeli
list_dir = [
    f"high_{prefix}_{i}" for prefix in prefixes for i in range(1, 6)
]

# Przetwarzanie plików w folderze
for directory in directories:
    for dir in list_dir:
        actual_file = directory + dir + "/metrics/"
        for file_name in sorted(os.listdir(actual_file)):
            if file_name.endswith(".csv") and file_name.startswith("best"):
                file_path = actual_file + file_name
                data = pd.read_csv(file_path)
                parsed_data = {
                    "AUC": eval(data["AUC"][0]),
                }

                # Dodawanie wartości do odpowiednich list
                for i, value in enumerate(parsed_data["AUC"]):
                    metrics["AUC"][i].append(float(value))

# Rysowanie boxplotów
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Przygotowanie danych dla boxplotów
for class_idx, (ax, class_name) in enumerate(zip(axes, classes)):
    data_to_plot = []
    labels = []

    for prefix_idx, prefix in enumerate(prefixes):
        start_idx = prefix_idx * 25  # 5 modeli na prefix
        end_idx = start_idx + 25
        data_to_plot.append(metrics["AUC"][class_idx][start_idx:end_idx])
        labels.append(f"{prefix[1:]}")

    # Rysowanie boxplotów
    ax.boxplot(
        data_to_plot,
        patch_artist=False,
        notch=False,
        showfliers=False
    )

    # Formatowanie wykresu
    ax.set_title(f"{class_name} - AUC")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUC")
    # ax.set_xlabel("% change filters")
    ax.grid(True)

# Dopasowanie układu
plt.tight_layout()
plt.show()

print(metrics)