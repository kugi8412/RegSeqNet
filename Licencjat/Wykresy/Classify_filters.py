import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PLIKI
folder_path = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/"
reference_file = folder_path + "/data/test_output/cohorts/reference/best000_highsignal_filter.txt"
stats_directory = folder_path + "data/test_output/cohorts/c1/stats/"

directories = [
    folder_path + "data/test_output/cohorts/c1/filter/comparison/",
    # folder_path + "data/test_output/cohorts/c2/filter/comparison/",
    # folder_path + "data/test_output/cohorts/c3/filter/comparison/",
    # folder_path + "data/test_output/cohorts/c4/filter/comparison/",
    # folder_path + "data/test_output/cohorts/c5/filter/comparison/",
    # folder_path + "data/test_output/cohorts/c6/filter/comparison/",
    # folder_path + "data/test_output/cohorts/c7/filter/comparison/",
]

# HISTOGRAMY
selected_distance = "L1-origin"

def classify_filter(mean, median):
    if mean > 1e-5 and median > 1e-5:
        return 'green'  # Kategoria 1
    elif mean > 1e-5:
        return 'orange'  # Kategoria 2
    else:
        return 'red'  # Kategoria 3

# Wczytanie statystyk
filter_colors = {}
for file in os.listdir(stats_directory):
    if file.endswith("stats.csv"):
        file_path = os.path.join(stats_directory, file)
        df_stats = pd.read_csv(file_path)
        
        for _, row in df_stats.iterrows():
            filter_name = row['Filter']
            mean_value = row['Mean']
            median_value = row['Median']
            
            filter_colors[filter_name] = classify_filter(mean_value, median_value)

# Rysowanie histogramu z kolorami
plt.figure(figsize=(10, 6))

for directory in directories:
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            
            if selected_distance not in df.columns:
                continue
            
            for _, row in df.iterrows():
                filter_name = row['Filtr']
                distance_value = row[selected_distance]
                color = filter_colors.get(filter_name, 'black')
                plt.hist(distance_value, bins=50, color=color, alpha=0.5, label=filter_name if filter_name not in plt.gca().get_legend_handles_labels()[1] else "")

plt.xlabel(selected_distance)
plt.ylabel("Filters counts")
plt.title("Histogram for specific case with categorized colors")
plt.legend()
plt.show()
