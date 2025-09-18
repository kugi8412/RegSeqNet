#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyse_filters(folder, reference_path, best_prefix='best', origin_prefix='origin'):
    """
    Analizuje wystąpienia filtrów w plikach 'best' i 'origin'.

    Parameters:
        folder (str): Ścieżka do folderu z plikami.
        reference_path (str): Ścieżka do pliku referencyjnego.
        best_prefix (str): Prefiks plików typu 'best'.
        origin_prefix (str): Prefiks plików typu 'origin'.

    Returns:
        pd.DataFrame: Wyniki analizy.
    """
    reference_data = pd.read_csv(reference_path)
    reference_filters = list(reference_data['Filter'])
    for i in range(len(reference_filters)):
        reference_filters[i] = int(reference_filters[i].split('_')[1])
    
    reference_filters = set(reference_filters)
    file_list = os.listdir(folder)

    # Split best and origin
    best_files = [f for f in file_list if ('below' not in f and f.startswith(best_prefix) and (f.endswith('filters_above_tresh.csv') or f.endswith(').csv')))]
    origin_files = [f for f in file_list if ('below' not in f and f.startswith(origin_prefix) and (f.endswith('filters_above_tresh.csv') or f.endswith(').csv')))]

    # Dict for results
    filter_counts = {}

    def process_files(file_list, category):
        for file_name in file_list:
            file_path = os.path.join(folder, file_name)
            data = pd.read_csv(file_path)
            for filter_name in data['Filter']:
                filter_name = filter_name.split('_')[1]
                filter_value = int(filter_name)
                if filter_name not in filter_counts:
                    filter_counts[str(filter_value)] = {'after_train': 0, 'before_train': 0, 'in_reference': False, 'files': []}
                filter_counts[filter_name][category] += 1
                filter_counts[filter_name]['files'].append(file_name.split('_')[0] + '_' + file_name.split('_')[2] + '_' + file_name.split('_')[3])

    process_files(best_files, 'after_train')
    process_files(origin_files, 'before_train')

    for filter_name in filter_counts:
        filter_counts[filter_name]['in_reference'] = int(filter_name) in reference_filters

    results = pd.DataFrame.from_dict(filter_counts, orient='index').reset_index()
    results.rename(columns={'index': 'filter'}, inplace=True)

    return results


def plot_frequency_analysis(results, start=0, end=300):
    results['total'] = results['after_train'] / results['before_train']
    top_results = results.sort_values(by='total', ascending=False).iloc[start:end]
    
    colors = ['red' if ref else 'skyblue' for ref in top_results['in_reference']]
    
    # Plot configuration
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(top_results['filter']))
    
    ax.bar(x, top_results['total'], color=colors)
    ax.set_ylabel('Occurrences (after_training / before_training)')
    ax.set_xlabel('Filters of the first layer')
    ax.set_title(f'Filters {start} to {end} by Frequency Ratio')
    ax.set_xticks(x)
    ax.set_xticklabels(top_results['filter'], rotation=45, ha='center', fontsize=8)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Reference Filters'),
        Patch(facecolor='skyblue', label='Other Filters')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_filter_analysis_bar_with_reference(results, start=100, end=200):
    results['total'] = results['after_train'] / results['before_train']
    top_results = results.sort_values(by='total', ascending=False).iloc[start:end]
    _, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(top_results['filter']))
    width = 0.35

    bars_best = ax.bar(x - width/2, top_results['after_train'], width, label='After Train', color='green', alpha=0.8)
    bars_origin = ax.bar(x + width/2, top_results['before_train'], width, label='Before Train', color='orange', alpha=0.8)

    # Mark reference files
    for i, ref in enumerate(top_results['in_reference']):
        if ref:
            ax.text(x[i], 0, '+', color='red', fontsize=8, fontweight='bold', ha='center')

    ax.set_ylabel('Occurrences')
    ax.set_xlabel('Filters of the first layer')
    ax.set_title(f'{start} - {end} Filters by frequency (+ alt_last before trainig)')
    ax.set_xticks(x)
    ax.set_xticklabels(top_results['filter'], rotation=45, ha='center', fontsize=8)
    ax.legend()

    plt.tight_layout()
    plt.show()

folder_path = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/"
reference_file = folder_path + 'statistics/Filtersplot/origin000_highsignal_filters_above_tresh.csv'

directory = folder_path + "/data/test_output/cohorts/c8/stats"

results = analyse_filters(directory, reference_file)
results['filter'] = results['filter'].astype('int')
results = results.sort_values(by='filter')

# print(results)

results.to_csv("filter_analysis_results.csv", index=False)

plot_frequency_analysis(results, 0, 100)

# Write results to CSV
output_csv_path = folder_path + "filter_analysis_results.csv"
results.to_csv(output_csv_path, index=False)
print(f"Results saved to {output_csv_path}")
