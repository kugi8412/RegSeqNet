#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_lost_filters_histogram(results):
    plt.figure(figsize=(16, 8))
    unique_cohorts = results['cohort'].unique()
    colors = plt.cm.tab10.colors[:len(unique_cohorts)]
    color_map = {cohort: colors[i] for i, cohort in enumerate(unique_cohorts)}
    
    # Labels by cohorts
    labels = [f"{row['cohort']}_{row['percentage']}%" for _, row in results.iterrows()]
    
    # Baarplot
    bars = []
    for i, (_, row) in enumerate(results.iterrows()):
        bar = plt.bar(i, row['lost'], color=color_map[row['cohort']])
        bars.append(bar[0])
    
    # Style
    plt.title('Number of filters reset against the alt model')
    plt.ylabel('Filter counts')
    plt.xlabel('Cohort_%Change')
    plt.xticks(range(len(results)), labels, rotation=90, ha='center', fontsize=8)
    
    # Write over bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=8)

    legend_elements = [plt.Rectangle((0,0),1,1, color=color_map[coh], label=coh) 
                      for coh in unique_cohorts]
    
    plt.legend(handles=legend_elements, title='Cohorts')
    plt.tight_layout()
    plt.show()


def plot_non_matching_histogram(results):
    plt.figure(figsize=(16, 8))

    unique_cohorts = results['cohort'].unique()
    colors = plt.cm.tab10.colors[:len(unique_cohorts)]
    color_map = {cohort: colors[i] for i, cohort in enumerate(unique_cohorts)}
    
    # Bars
    bars = []
    for i, (_, row) in enumerate(results.iterrows()):
        bar = plt.bar(i, row['non_matching'], color=color_map[row['cohort']])
        bars.append(bar[0])

    legend_elements = [plt.Rectangle((0,0),1,1, color=color_map[coh], label=coh) 
                      for coh in unique_cohorts]
    
    # Legend
    labels = [f"{row['cohort']}_{row['percentage']}%" for _, row in results.iterrows()]
    # bbox_to_anchor=(1.05, 1)
    plt.legend(handles=legend_elements, title='Cohorts')
    plt.title('Filters not in the alt model')
    plt.ylabel('Filters Counts')
    plt.xlabel('Cohort_%Change')
    plt.xticks(range(len(results)), labels, rotation=90, ha='center', fontsize=8)

    plt.tight_layout()
    plt.show()


def compare_filters(directories, reference_path, new_files):
    # Reference files
    reference_data = pd.read_csv(reference_path)
    reference_filters = set(reference_data['Filter'])
    results = []

    for directory in directories:
        cohort = directory[-9:-7]
        if cohort == "10":
            cohort = 'c_10'

        for file_name in sorted(os.listdir(directory)):
            if file_name.startswith("best") and file_name.endswith("filters_above_tresh.csv"):
                file_path = os.path.join(directory, file_name)
                comparison_data = pd.read_csv(file_path)
                comparison_filters = set(comparison_data['Filter'])
                matching = len(reference_filters & comparison_filters)
                non_matching = len(comparison_filters - reference_filters)
                lost = len(reference_filters - comparison_filters)
                
                # Split name of model
                try:
                    percentage = int(file_name.split('_')[2])
                except:
                    print(file_name)

                label = f"{percentage}_{file_name.split('_')[3]}_{cohort}"
                
                # Compare with references
                origin_file = file_name.replace("best", "origin")
                origin_path = os.path.join(directory, origin_file)
                origin_data = pd.read_csv(origin_path)
                origin_filters = set(origin_data['Filter'])
                origin = len(origin_filters - comparison_filters)
                
                results.append({
                    'cohort': cohort,
                    'percentage': percentage,
                    'label': label,
                    'matching': matching,
                    'non_matching': non_matching,
                    'origin': origin,
                    'lost': lost
                })

    # outliers
    for i in range(0, len(new_files), 2):
        file = new_files[i]
        comparison_data = pd.read_csv(file)
        comparison_filters = set(comparison_data['Filter'])
        matching = len(reference_filters & comparison_filters)
        non_matching = len(comparison_filters - reference_filters)
        origin_data = pd.read_csv(new_files[i+1])
        origin_filters = set(origin_data['Filter'])
        origin = len(origin_filters - comparison_filters)
        lost = len(reference_filters - comparison_filters)

        if '100' in file:
            results.append({
                'cohort': 0,
                'percentage': 100,
                'label': '100',
                'matching': matching,
                'non_matching': non_matching,
                'origin' : origin,
                'lost': lost
            })
        else:
            results.insert(0, {
                'cohort': 0,
                'percentage': 0,
                'label': '0',
                'matching': matching,
                'non_matching': non_matching,
                'origin' : origin,
                'lost': lost
            })

    df = pd.DataFrame(results)
    df = df.sort_values(by=['percentage', 'cohort'])
    return df

def plot_comparison_results(results):
    x = np.arange(len(results))
    width = 0.6
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Histogram
    ax.bar(x, results['matching'], width,
           label='Matching Filters with alt after training', color='#2ca02c')
    ax.bar(x, results['non_matching'], width, bottom=results['matching'],
           label='Non-Matching Filters with alt after training', color='#ff7f0e')
    ax.bar(x, results['origin'], width, 
           bottom=results['matching']+results['non_matching'], 
           label='After change model', color='#d62728')

    # Legend
    ax.set_ylabel('Number of Filters')
    ax.set_xlabel('%Change_Model_Cohort')
    ax.set_title('Filters number comparison before and after training on Highsignal dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(results['label'], rotation=90, ha='center', fontsize=8)
    ax.legend(title='Filter types', loc='upper left')
    plt.tight_layout()
    plt.show()


folder_path = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/"
reference_file = folder_path + 'statistics/Filtersplot/origin000_highsignal_filters_above_tresh.csv'

directories = [
    # folder_path + "data/test_output/cohorts/c1/stats/",
    folder_path + "data/test_output/cohorts/c2/stats/",
    folder_path + "data/test_output/cohorts/c3/stats/",
    folder_path + "data/test_output/cohorts/c4/stats/",
    folder_path + "data/test_output/cohorts/c5/stats/",
    folder_path + "data/test_output/cohorts/c6/stats/",
    folder_path + "data/test_output/cohorts/c7/stats/",
    folder_path + "data/test_output/cohorts/c8/stats/",
    folder_path + "data/test_output/cohorts/c9/stats/",
    folder_path + "data/test_output/cohorts/c10/stats/"
]

new_files = [
    folder_path + 'statistics/Filtersplot/best000_highsignal_filters_above_tresh.csv',
    folder_path + 'statistics/Filtersplot/origin000_highsignal_filters_above_tresh.csv', 
    folder_path + 'statistics/Filtersplot/best100_highsignal_filters_above_tresh.csv',
    folder_path + 'statistics/Filtersplot/origin100_highsignal_filters_above_tresh.csv',
]


results = compare_filters(directories, reference_file, new_files)

# plot_comparison_results(results)
# plot_lost_filters_histogram(results)
plot_non_matching_histogram(results)
