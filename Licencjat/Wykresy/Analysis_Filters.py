import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PLIKI
folder_path = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/"
reference_file = folder_path + "/data/test_output/cohorts/reference/best000_highsignal_filter.txt"

directories = [
    folder_path + "data/test_output/cohorts/c1/filter/",
    folder_path + "data/test_output/cohorts/c2/filter/",
    folder_path + "data/test_output/cohorts/c3/filter/",
    folder_path + "data/test_output/cohorts/c4/filter/",
    folder_path + "data/test_output/cohorts/c5/filter/",
    folder_path + "data/test_output/cohorts/c6/filter/",
    folder_path + "data/test_output/cohorts/c7/filter/"
]
directories = [
    folder_path + "data/test_output/cohorts/c8/filter/",
]

######### TO CHANGE ########
#directories = [
#    folder_path + "data/test_output/cohorts/reference/",
#]

# Odczytujemy filtry
def load_filters(file_path):
    filters = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        current_filter = None
        for line in lines:
            line = line.strip()
            if line.startswith('>filter_'):
                current_filter = line[1:]
                filters[current_filter] = []
            elif current_filter is not None:
                filters[current_filter].append(list(map(float, line.split())))
    return {k: np.array(v) for k, v in filters.items()}

# Liczymy odległośc normy
def compute_metrics(matrix1, matrix2):
    l1_distance = np.sum(np.abs(matrix1 - matrix2))
    l2_distance = np.linalg.norm(matrix1 - matrix2, ord='fro')
    return l1_distance, l2_distance

# PLIK REFERENCYJNY
reference_filters = load_filters(reference_file)

# Przechodzimy po folderach
def process_directory(directory, special_filters=['filter_11', 'filter_18', 'filter_23', 'filter_27', 'filter_30',
                                                  'filter_48', 'filter_50', 'filter_53', 'filter_65', 'filter_70',
                                                  'filter_71', 'filter_76', 'filter_103', 'filter_104', 'filter_105',
                                                  'filter_108', 'filter_113', 'filter_121', 'filter_136', 'filter_138',
                                                  'filter_141', 'filter_148', 'filter_153', 'filter_158', 'filter_159',
                                                  'filter_167', 'filter_182', 'filter_188', 'filter_191', 'filter_193',
                                                  'filter_200', 'filter_203', 'filter_205', 'filter_214', 'filter_216',
                                                  'filter_221', 'filter_225', 'filter_229', 'filter_247', 'filter_267',
                                                  'filter_285', 'filter_288', 'filter_292', 'filter_299',
                                                  'filter_17', 'filter_236']):
    comparison_folder = os.path.join(directory, "comparison")
    os.makedirs(comparison_folder, exist_ok=True)

    # Znalezienie par plików
    origin_files = {f: os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("origin")}
    best_files = {f.replace("origin", "best"): os.path.join(directory, f.replace("origin", "best")) for f in origin_files}
    
    for best_name, best_path in best_files.items():
        origin_name = best_name.replace("best", "origin")
        if origin_name not in origin_files:
            continue
        
        origin_filters = load_filters(origin_files[origin_name])
        best_filters = load_filters(best_path)

        data = []
        for filter_name in best_filters:
            # Obliczanie norm dla origin
            if filter_name in origin_filters:
                l1_origin, l2_origin = compute_metrics(origin_filters[filter_name], np.zeros_like(origin_filters[filter_name]))
            else:
                l1_origin, l2_origin = np.nan, np.nan
            
            # Obliczanie norm dla best
            l1_best, l2_best = compute_metrics(best_filters[filter_name], np.zeros_like(best_filters[filter_name]))

            # Obliczanie metryk względem origin
            if filter_name in origin_filters:
                l1_origin_diff, l2_origin_diff = compute_metrics(best_filters[filter_name], origin_filters[filter_name])
            else:
                l1_origin_diff, l2_origin_diff = np.nan, np.nan
            
            # Obliczanie metryk względem reference
            if filter_name in reference_filters:
                l1_ref, l2_ref = compute_metrics(best_filters[filter_name], reference_filters[filter_name])
            else:
                l1_ref, l2_ref = np.nan, np.nan
            
            near_zero_count = np.sum(np.abs(best_filters[filter_name]) < 1e-5)

            special_filter = filter_name in special_filters
            
            data.append([
                filter_name, 
                l1_origin, l2_origin, 
                l1_best, l2_best, 
                l1_origin_diff, l2_origin_diff, 
                l1_ref, l2_ref, 
                special_filter, 
                near_zero_count
            ])
        
        df = pd.DataFrame(data, columns=[
            "Filtr", 
            "origin-L1", "origin-L2", # original values
            "best-L1", "best-L2", # best values
            "L1-origin", "L2-origin", 
            "L1-ref", "L2-ref", 
            "AboveRef", 
            "Count_0"
        ])
        
        df.to_csv(os.path.join(comparison_folder, best_name.replace(".txt", ".csv")), index=False)


for dir_path in directories:
    process_directory(dir_path)

print("Analiza zakończona!")

directories = [
    # folder_path + "data/test_output/cohorts/c1/filter/comparison/",
    folder_path + "data/test_output/cohorts/c2/filter/comparison/",
    folder_path + "data/test_output/cohorts/c3/filter/comparison/",
    folder_path + "data/test_output/cohorts/c4/filter/comparison/",
    folder_path + "data/test_output/cohorts/c5/filter/comparison/",
    folder_path + "data/test_output/cohorts/c6/filter/comparison/",
    folder_path + "data/test_output/cohorts/c7/filter/comparison/",
]
'''
directories = [
    folder_path + "data/test_output/cohorts/reference/comparison",
]
'''
directories = [
    folder_path + "data/test_output/cohorts/c8/filter/comparison/",
    # folder_path + "data/test_output/cohorts/c9/filter/comparison/",
]
# HISTOGRAMY
selected_distance = "L2-ref"

special_filters_values = []
other_filters_values = []
sparse_filters_values = []

# Przetwarzanie plików CSV w folderach
for directory in directories:
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_path = os.path.join(directory, file)

            # Wczytanie pliku CSV
            df = pd.read_csv(file_path)

            # Sprawdzenie, czy wymagane kolumny istnieją
            if selected_distance not in df.columns or "AboveRef" not in df.columns or "Count_0" not in df.columns:
                print(f"Pomijam {file}")
                continue

            # Klasyfikacja wartości
            special_filters_values.extend(df[(df["AboveRef"] == True)][selected_distance])
            other_filters_values.extend(df[(df["AboveRef"] == False) & (df["Count_0"] == 0)][selected_distance])
            sparse_filters_values.extend(df[((df["AboveRef"] == True) & (df["Count_0"] >= 1) & (df["Count_0"] < 78)) | 
                                            ((df["AboveRef"] == False) & (df["Count_0"] >= 1) & (df["Count_0"] < 78))][selected_distance])

# Filtrowanie wartości powyżej 0.05 dla innych filtrów
filtered_other_filters_values = [val for val in other_filters_values if val > 0.05]
filtered_sparse_filters_values = [val for val in sparse_filters_values if val > 0.05]

# Rysowanie zbiorczego histogramu
plt.figure(figsize=(10, 6))
plt.hist(special_filters_values, bins=50, color='blue', alpha=0.5, label='Special filters')
plt.hist(filtered_other_filters_values, bins=50, color='red', alpha=0.5, label='Other filters')
plt.hist(filtered_sparse_filters_values, bins=50, color='green', alpha=0.5, label='Sparse filters')

plt.xlabel(selected_distance)
plt.ylabel("Filters counts")
plt.title("Histogram for specific case")
plt.legend()
plt.show()


'''
folder_path = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/"
    base_dir = folder_path + "/data/test_output/cohorts/"
    selected_cohorts = ['c8', 'c9']
    thresholds_avg = [0, 0.01, 0.5]
    top_range = (50, 100)  # filtry 50-100
    special_filters = np.array(['11', '18', '23', '27', '30',
                        '48', '50', '53', '65', '70',
                        '71', '76', '103', '104', '105',
                        '108', '113', '121', '136', '138',
                        '141', '148', '153', '158', '159',
                        '167', '182', '188', '191', '193',
                        '200', '203', '205', '214', '216',
                        '221', '225', '229', '247', '267',
                        '285', '288', '292', '299',
                        '17', '236'], dtype='int64)  # przykładowo

                        
# Wykres średnich norm z etykietami
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(sel))
        colors = ['red' if idx in special_filters else 'blue' for idx in sel]
        bars = ax.bar(x, values, color=colors)
        ax.set_xlabel('Filter Index (global idx)')
        ax.set_ylabel('Avg L2 Norm')
        ax.set_title(f'Filters {start+1}-{end}, avg L2 > {thr}')
        ax.set_xticks(x)
        ax.set_xticklabels([str(idx) for idx in sel], rotation=90)
        # dodaj etykiety nad słupkami
        for bar, idx in zip(bars, sel):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height*1.01, str(idx),
                    ha='center', va='bottom', fontsize=8)
        # Legenda
        ax.plot([], [], color='red', label='Special filters')
        ax.plot([], [], color='blue', label='Other filters')
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{output_dir}\\avg_{start+1}_{end}_thr_{thr}.png")
        plt.close(fig)
'''
