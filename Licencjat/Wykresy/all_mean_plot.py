import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Folder
folder_path = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/"
directories = [
    folder_path + "data/test_output/cohorts/c1/high/",
    folder_path + "data/test_output/cohorts/c4/high/",
    folder_path + "data/test_output/cohorts/c2/high/",
]
directories = [
    folder_path + "data/test_output/cohorts/c8/high/",
]
folder_path += "test_set/best_high/prediction/highsignal/metrics"

# Inicjalizacja przechowywania danych
list_cohorts = ["c8", "c9", "c10"]
model_means = []  # Średnie wartości AUC dla każdego modelu
x_labels = []

# Lista katalogów modeli
list_dir = [
    f"high_{prefix}_{i}" for prefix in ["020", "040", "060", "080"] for i in range(1, 6)
]

# Przetwarzanie plików w folderach
for cohort_idx, directory in enumerate(directories):
    for model_idx, dir in enumerate(list_dir):
        actual_file = os.path.join(directory, dir, "metrics")
        auc_values = []
        
        for file_name in sorted(os.listdir(actual_file)):
            if file_name.endswith(".csv") and file_name.startswith("best"):
                file_path = os.path.join(actual_file, file_name)
                data = pd.read_csv(file_path)
                auc_values.extend(eval(data["AUC"][0]))
        
        # Obliczenie średniej AUC dla modelu
        if auc_values:
            model_means.append(np.mean(auc_values))
        else:
            model_means.append(np.nan)
        
        x_labels.append(f"{dir[6:]}_{list_cohorts[cohort_idx]}")

# Wczytanie plików referencyjnych
comparison_files = ["alt1_last_metrics.csv", "best000_highsignal_metrics.csv"]
comparison_means = []
# models = ["alt_last", "high_last"]
models = ["alt_before", "alf_after"]

for comparison_file in comparison_files:
    file_path = os.path.join(folder_path, comparison_file)
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        auc_values = [float(value) for value in eval(data["AUC"][0])]
        comparison_means.append(np.mean(auc_values))

new_files = ["best100_highsignal_metrics.csv"]
for new_file in new_files:
    file_path = os.path.join(folder_path, new_file)
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        auc_values = [float(value) for value in eval(data["AUC"][0])]
        if '100' in new_file:
            model_means.append(np.mean(auc_values))
            x_labels.append("100")
        else:
            model_means.insert(0, np.mean(auc_values))
            x_labels.insert(0, "0")

# Tworzenie wykresu
plt.figure(figsize=(13, 8))

# Rysowanie średnich AUC dla modeli
plt.scatter(
    np.arange(len(x_labels)),
    model_means,
    color="blue",
    label="Mean AUC"
)

# Rysowanie linii referencyjnych
line_styles = ["--", "-"]
for idx, reference_value in enumerate(comparison_means):
    plt.axhline(
        reference_value,
        color="red",
        linestyle=line_styles[idx],
        linewidth=1.2,
        label=f"{models[idx]}"
    )

plt.title("Mean AUC on Highsignal dataset for all models")
plt.xlabel("%Change_Model_Cohort")
plt.ylabel("Mean AUC")
plt.grid(True)
plt.legend()

# Ustawienie etykiet osi X
plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45, ha="right", fontsize=6)

plt.tight_layout()
plt.show()
