import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Folder (zmiana w 2 miejscach)
folder = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/"
directory = folder + "data/test_output/cohorts/c4/alt/"
folder_path = folder + "test_set/best_alt/prediction/highsignal/metrics"


# Inicjalizacja przechowywania danych
best_metrics = {
    "AUC": [],
    "Sensitivity": [],
    "Specificity": [],
    "Losses": [],
}
origin_metrics = {
    "AUC": [],
    "Sensitivity": [],
    "Specificity": [],
    "Losses": [],
}
comparison_metrics = {
    "AUC": [[], []],
    "Sensitivity": [[], []],
    "Specificity": [[], []],
    "Losses": [[], []],
}
best_values = []
origin_values = []

# Pliki do porównania
comparison_files = ["alt1_last_metrics.csv", "high_signal4_last_metrics.csv"]
models = ["alt_last", "high_last"]
list_dir = ["high_020_1", "high_020_2", "high_020_3", "high_020_4", "high_020_5",
            "high_040_1", "high_040_2", "high_040_3", "high_040_4", "high_040_5",
            "high_060_1", "high_060_2", "high_060_3", "high_060_4", "high_060_5",
            "high_080_1", "high_080_2", "high_080_3", "high_080_4", "high_080_5"
            ]
labels_model = [model[6:] for model in list_dir]

# Przetwarzanie plików w kolejnych folderach
for dir in list_dir:
    actual_file = directory + dir + "/metrics/"
    for file_name in sorted(os.listdir(actual_file)):
        # print(file_name)
        if not file_name.endswith("diff.csv"):
            file_path = actual_file + file_name
            file = file_name.split('_')[2] + '_' + file_name.split('_')[3]
            data = pd.read_csv(file_path)
            parsed_data = {
                    "AUC": eval(data["AUC"][0]),
                    "Sensitivity": eval(data["Sensitivity"][0]),
                    "Specificity": eval(data["Specificity"][0]),
                    "Losses": eval(data["Losses"][0]),
                }
            if file_name.startswith("best"):
                best_values.append(int(file[0:3]) + int(file[4:]) * 4)
                for metric, values in parsed_data.items():
                    best_metrics[metric].append(np.mean(values))
            elif file_name.startswith("origin"):
                origin_values.append(int(file[0:3]) + int(file[4:]) * 4)
                for metric, values in parsed_data.items():
                    origin_metrics[metric].append(np.mean(values))

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
        # Dodawanie średnich wartości dla każdego pliku referencyjnego
        for metric, values in parsed_data.items():
            comparison_metrics[metric][idx] = np.mean(values)

# Styl dla plików referencyjnych
class_colors = ["C1", "C0"]
file_line_styles = ["--", "-"]

# Tworzenie wykresów
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Tytuły wykresów
titles = ["AUC", "Sensitivity", "Specificity", "Losses"]

# Rysowanie wykresów
for ax, (metric, values) in zip(axes.flatten(), best_metrics.items()):
    # Średnie wartości dla danych testowych
    ax.plot(
        best_values,
        values,
        color="green",
        marker="o",
        linestyle="",
        label=f"After train"
    )

for ax, (metric, values) in zip(axes.flatten(), origin_metrics.items()):
    # Średnie wartości dla danych testowych
    ax.plot(
        best_values,
        values,
        color="red",
        marker=".",
        linestyle="",
        label=f"Before train"
    )
        # Linie poziome dla średnich wartości z plików referencyjnych
    for idx, mean_value in enumerate(comparison_metrics[metric]):
        ax.axhline(
            mean_value,
            color=class_colors[idx],
            linestyle=file_line_styles[idx],
            linewidth=1.5,
            label=f"{models[idx]}"
        )
    ax.set_title(metric)
    ax.set_xlabel("% of changed filters")
    ax.set_ylabel("Mean " + metric)
    ax.grid(True)
    ax.legend()
    ax.set_xticks(best_values)
    ax.set_xticklabels(labels_model, rotation=45)


# Dopasowanie układu
plt.tight_layout()
plt.show()
