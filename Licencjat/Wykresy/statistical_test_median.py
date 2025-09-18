import os
import pandas as pd
import scipy.stats as stats
import itertools
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
metrics = {"AUC": {}, "MCC": {}}
prefixes = ["020", "040", "060", "080"]
classes = ["PA", "NA", "PI", "NI"]

# Inicjalizacja słowników dla każdej klasy i kohorty
for metric in metrics.keys():
    for class_name in classes:
        metrics[metric][class_name] = {prefix: [] for prefix in prefixes}

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
                    "MCC": eval(data["MCC"][0])
                }

                for i, class_name in enumerate(classes):
                    prefix = dir.split("_")[1]
                    metrics["AUC"][class_name][prefix].append(float(parsed_data["AUC"][i]))
                    metrics["MCC"][class_name][prefix].append(float(parsed_data["MCC"][i]))

# Wyniki
results = []

# Analiza dla AUC i MCC
for metric_name in ["AUC", "MCC"]:
    for class_name in classes:
        # Dla każdej miary, dla klasy porównanie kohort
        data_cohorts = metrics[metric_name][class_name]

        # Test Kruskala-Wallisa
        stat, p_value = stats.kruskal(*data_cohorts.values())
        results.append([metric_name, class_name, "All Samples", "Kruskal-Wallis", "All", "All", stat, p_value])

        # Test Friedmana
        try:
            friedman_stat, friedman_p = stats.friedmanchisquare(*data_cohorts.values())
            results.append([metric_name, class_name, "All Samples", "Friedman", "All", "All", friedman_stat, friedman_p])
        except ValueError:
            results.append([metric_name, class_name, "All Samples", "Friedman", "All", "All", "Error", "Not enough paired samples"])

        # Wybieramy pary
        for (cohort1, data1), (cohort2, data2) in itertools.combinations(data_cohorts.items(), 2):
            # Test U Manna-Whitneya
            u_stat, u_p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            results.append([metric_name, class_name, f"{cohort1} vs {cohort2}", "U Mann-Whitney", cohort1, cohort2, u_stat, u_p_value])

            # Test Wilcoxona
            try:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(data1[:len(data1)], 
                                            data2[:len(data2)])
                results.append([metric_name, class_name, f"{cohort1} vs {cohort2}", "Wilcoxon", cohort1, cohort2, wilcoxon_stat, wilcoxon_p])
            except ValueError:
                results.append([metric_name, class_name, f"{cohort1} vs {cohort2}", "Wilcoxon", cohort1, cohort2, "Error", "Unequal sample sizes"])

# Zapisanie pliku
output_df = pd.DataFrame(results, columns=["Metric", "Class", "Comparison", "Test", "Group 1", "Group 2", "Statistic", "P-Value"])
output_df.to_csv(folder_path + "/statistical_tests_between_cohorts.csv", index=False)

print("OK")
print(output_df["P-Value"].min())
