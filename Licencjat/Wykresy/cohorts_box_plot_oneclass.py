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
fig, ax = plt.subplots(figsize=(10, 8))

# Przygotowanie danych dla boxplotów
data_to_plot = []
positions = []
labels = []

for prefix_idx, prefix in enumerate(prefixes):
    for class_idx, class_name in enumerate(classes):
        start_idx = prefix_idx * 25  # 25 modeli na prefix
        end_idx = start_idx + 25
        data_to_plot.append(metrics["AUC"][class_idx][start_idx:end_idx])
        positions.append(prefix_idx * 5 + class_idx + 1)
        labels.append(f"{class_name}-{prefix[1:]}")

# Rysowanie boxplotów
box = ax.boxplot(
    data_to_plot,
    positions=positions,
    patch_artist=True,
    notch=False,
    showfliers=False
)

# Dodanie punktów dla każdego modelu
for i, class_data in enumerate(data_to_plot):
    x = [positions[i]] * len(class_data)
    ax.plot(x, class_data, 'o', color='black', alpha=0.8)

# Formatowanie wykresu
ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_title("AUC boxplot for each class")
ax.set_ylabel("AUC")
ax.grid(True)

# Dodanie kolorów dla grup
colors = ["lightblue", "lightgreen", "lightpink", "lightgray"]
for patch, color in zip(box['boxes'], colors * len(prefixes)):
    patch.set_facecolor(color)

# Wyświetlenie wykresu
plt.tight_layout()
plt.show()

print(metrics)