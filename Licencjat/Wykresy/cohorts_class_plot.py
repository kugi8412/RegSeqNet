import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Folder (zmiana w 2 miejscach)
folder_path = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/"
directory = folder_path + "data/test_output/cohorts/c8/high/"
folder_path += "test_set/best_high/prediction/highsignal/metrics"

# Inicjalizacja przechowywania danych
metrics = {
    "AUC": [[] for _ in range(4)],
    "Sensitivity": [[] for _ in range(4)],
    "Specificity": [[] for _ in range(4)],
    "Losses": [[] for _ in range(4)],
}
x_values = []

comparison_metrics = {
    "AUC": [[], []],  # Dwie listy dla dwóch plików do porównania
    "Sensitivity": [[], []],
    "Specificity": [[], []],
    "Losses": [[], []],
}

# Pliki do porównania (ustal ich nazwy ręcznie)
comparison_files = ["alt1_last_metrics.csv", "high_signal4_last_metrics.csv"]
models = ["alt_last", "high_last"]
list_dir = ["high_020_1", "high_020_2", "high_020_3", "high_020_4", "high_020_5",
            "high_040_1", "high_040_2", "high_040_3", "high_040_4", "high_040_5",
            "high_060_1", "high_060_2", "high_060_3", "high_060_4", "high_060_5",
            "high_080_1", "high_080_2", "high_080_3", "high_080_4", "high_080_5"
            ]
labels_model = [model[6:] for model in list_dir]

# Przetwarzanie plików w folderze
for dir in list_dir:
    actual_file = directory + dir + "/metrics/"
    for file_name in sorted(os.listdir(actual_file)):
        if file_name.endswith(".csv") and file_name.startswith("best"):
            file = file_name.split('_')[2] + '_' + file_name.split('_')[3]
            file_path = actual_file + file_name
            x_values.append(int(file[0:3]) + int(file[4:]) * 4)
            data = pd.read_csv(file_path)
            parsed_data = {
                "AUC": eval(data["AUC"][0]),
                "Sensitivity": eval(data["Sensitivity"][0]),
                "Specificity": eval(data["Specificity"][0]),
                "Losses": eval(data["Losses"][0]),
            }

            # Dodawanie wartości do odpowiednich list
            for metric, values in parsed_data.items():
                for i, value in enumerate(values):
                    metrics[metric][i].append(float(value))

# Przetwarzanie plików porównawczych
for idx, comparison_file in enumerate(comparison_files):
    file_path = os.path.join(folder_path, comparison_file)
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        parsed_data = {
            "AUC": eval(data["AUC"][0]),
            "Sensitivity": eval(data["Sensitivity"][0]),
            "Specificity": eval(data["Specificity"][0]),
            "Losses": eval(data["Losses"][0]),
        }
        # Dodawanie wartości dla każdej klasy
        for metric, values in parsed_data.items():
            comparison_metrics[metric][idx] = [float(value) for value in values]

# Definicja kolorów i stylów linii
class_colors = ["C0", "C1", "C2", "C3"]  # Kolory dla klas
file_line_styles = ["--", "-"]  # Style linii dla plików referencyjnych
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
titles = ["AUC", "Sensitivity", "Specificity", "Losses"]

# Rysowanie wykresów
classes = ["PA", "NA", "PI", "NI"]
for ax, (metric, values) in zip(axes.flatten(), metrics.items()):
    for i, class_values in enumerate(values):
        ax.plot(x_values, class_values, marker="o", color=class_colors[i], label=f"{classes[i]}", linestyle="None")
    
    for idx, comparison_values in enumerate(comparison_metrics[metric]):
        for class_idx, value in enumerate(comparison_values):
            ax.axhline(
                value,
                color=class_colors[class_idx],
                linestyle=file_line_styles[idx],
                linewidth=1.0,
                label=f"{models[idx]}" if class_idx == 0 else None
            )
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.005)
    ax.set_title(metric)
    ax.set_xlabel("% of changed filters")
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.legend()
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels_model, rotation=45)

# Dopasowanie układu
plt.tight_layout()
plt.show()
