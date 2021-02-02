import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import math
from bin.common import *

parser = argparse.ArgumentParser(description='Plot results based on given table')
parser.add_argument('-t', '--table', action='store', metavar='NAME', type=str, default=None,
                    help='Results table with data to plot, if PATH is given, file is supposed to be '
                         'in PATH directory: [PATH]/[NAME], default: [PATH]/[NAMESPACE]_train/test_results.tsv')
parser = basic_params(parser, param=True)
parser.add_argument('-c', '--column', action='store', metavar='COL', nargs='+', type=str, default=['loss'],
                    help='Number or name of column(s) to plot, default: loss')
group1 = parser.add_mutually_exclusive_group(required=False)
group1.add_argument('--train', action='store_true',
                    help='Use values from training, default values from validation are used')
group1.add_argument('--test', action='store_true',
                    help='Use testing results.')
group1.add_argument('--cv', action='store_true',
                    help='Use CV results.')
parser.add_argument('--not_valid', action='store_true',
                    help='Do not print values from validation')
parser.add_argument('--print_mean', action='store_true',
                    help='Print also mean of the given data')
parser.add_argument('--plot_one', action='store_true',
                    help='Plot only AUC for given neuron vs rest (not given neuron vs other neuron)')
parser.add_argument('--subset', action='store_true',
                    help='Plot results for the subset of sequences (from [PATH]/[NAMESPACE]_subset-train_results.tsv)')
group2 = parser.add_mutually_exclusive_group(required=False)
group2.add_argument('--scatter', action='store_true',
                    help='Scatter plot')
group2.add_argument('--boxplot', action='store_true',
                    help='Boxplot plot of values')
args = parser.parse_args()


path, output, namespace, seed = parse_arguments(args, args.table, model_path=True)

train = False
valid = True
test = False
cv = False
all_ = False
if args.train:
    train = True
elif args.test:
    test = True
    valid = False
elif args.cv:
    cv = True
    valid = False
if args.not_valid:
    valid = False

if args.boxplot:
    boxplot = True
    scatter = False
else:
    scatter = True
    boxplot = False

if args.table is not None:
    if args.path is not None:
        table = os.path.join(args.path, args.table)
    else:
        table = args.table
elif test:
    if args.subset:
        table = os.path.join(path, namespace + '_subset-test_results.tsv')
    else:
        table = os.path.join(path, namespace + '_test_results.tsv')
elif train or valid:
    if args.subset:
        table = os.path.join(path, namespace + '_subset-train_results.tsv')
    else:
        table = os.path.join(path, namespace + '_train_results.tsv')
elif cv:
    table = os.path.join(path, namespace + '_cv_results.tsv')
else:
    table = ''
if not os.path.isfile(table):
    table = os.path.join(path, namespace + '_results.tsv')

if args.param is not None:
    if args.path is not None:
        param = os.path.join(path, args.param)
    else:
        param = args.param
else:
    param = os.path.join(path, namespace + '_params.txt')

columns = args.column

epoch = -1
epochs = []
xticks = []
with open(table, 'r') as f:
    header = f.readline().strip().split('\t')
    colnum = []
    for c in columns:
        if str.isdigit(c):
            colnum.append(int(c) - 1)
        else:
            try:
                colnum.append(header.index(SHORTCUTS[c]))
            except ValueError:
                colnum += [i for i, el in enumerate(header) if SHORTCUTS[c] in el]
    if test:
        stages = ['all']
    elif cv:
        stages = ['cv']
    else:
        stages = [el for el in STAGES.keys() if globals()[el]]
        if args.subset:
            stages = ['subset_{}'.format(el) for el in stages]
    values = [[[] for _ in colnum] for el in stages]  # for each stage and for each column
    for e, line in enumerate(f):
        line = line.strip().split('\t')
        if train or valid:
            if int(line[0]) > epoch:
                epoch = int(line[0])
                epochs.append(epoch)
            elif int(line[0]) < epoch:
                raise ValueError
            if line[1] in stages:
                i = stages.index(line[1])
                for j, c in enumerate(colnum):
                    values[i][j].append([float(el) if el not in ['-', 'None', 'nan'] else np.nan for el in line[c].split(', ')])
        elif test or cv:
            epochs.append(e)
            xticks.append('{}-{}'.format(os.path.split(line[0])[1], line[1]))
            for j, c in enumerate(colnum):
                values[0][j].append([float(el) if el not in ['-', 'None', 'nan'] else np.nan for el in line[c].split(', ')])

neurons = get_classes_names(param)
colors = {}
for n, c in zip(neurons, COLORS):
    colors[n] = c
if args.plot_one:
    to_del = {}
    for i, (stage, value) in enumerate(zip(stages, values)):
        for j, c in enumerate(colnum):
            y = [el[j] for el in value[j]]
            if all([math.isnan(el) for el in y]):
                to_del[j] = to_del.setdefault(j, []) + [i]
    num_neurons = len(neurons)
    for k, v in to_del.items():
        if len(v) == len(stages):
            for i in v:
                dif_num_neurons = num_neurons - len(values[i])
                del values[i][k - dif_num_neurons]
                for el, vv in enumerate(values[i]):
                    for la, _ in enumerate(vv):
                        del values[i][el][la][k - dif_num_neurons]
            dif_num_neurons = num_neurons - len(colnum)
            if 'auc' in args.column:
                del colnum[k - dif_num_neurons]
            del neurons[k - dif_num_neurons]

try:
    values = np.nan_to_num(values)
    if 'auc' in args.column:
        ylims = [0, 1]
    else:
        ylims = [1, 1.55]
        #notzero_values = [el for el in values.flatten() if el != 0]
        #ylims = [np.min(notzero_values) - 0.05, np.max(notzero_values) + 0.05]
except ValueError:
    print('No values were read from the results file!')
    raise ValueError


def plot_one(ax, x, y, line, label, color):
    if not all([el == 0 for el in y]):
        ax.plot(x, y, line, label=label, alpha=0.5, color=color)
        # ax.set_xlabel('Epoch')
        ax.set_ylim(*ylims)


if cv:
    colnum = colnum[:1]
fig, axes = plt.subplots(nrows=len(colnum), ncols=len(stages), figsize=(12, 8), squeeze=False, sharex=True, sharey=True)
if axes.shape[1] > 1:
    num_xticks = 6
else:
    num_xticks = 10
for i, (stage, value) in enumerate(zip(stages, values)):  # for each stage
    if args.subset:
        title = STAGES[stage.replace('subset_', '')] + ' - subset'
    else:
        title = STAGES[stage]
    axes[0, i].set_title(title)
    axes[-1, i].set_xlabel('Epoch')
    for j, c in enumerate(colnum):  # for each column
        a = axes[j][i]
        for side in ['right', 'left', 'top', 'bottom']:
            a.spines[side].set_visible(False)
        a.set_facecolor('#E3E3E3')
        if boxplot:
            if cv:
                y = [el[0] for el in value]
                a.set_ylabel(header[c].split('-')[0])
            else:
                y = [[el[k] for el in value[j]] for k in range(len(neurons))]
                a.set_ylabel(header[c].replace('-', '-\n'))
            a.boxplot(y, showmeans=True)
            a.set_xticklabels(neurons)
        elif scatter:
            if i == 0:
                color = 'black'
                for n in neurons:
                    if n in header[c]:
                        color = colors[n]
                if 'auc' in args.column and (args.subset or '1561' in namespace) and \
                        (('gb-positive' in namespace and j == 0) or ('pa-da-positive' in namespace and j == 1)):
                    ytitle = header[c].replace('-', '-\n') + '\n(GB specific sequences)'
                elif 'auc' in args.column and (args.subset or '1561' in namespace) and \
                        (('gb-positive' in namespace and j == 1) or ('pa-da-positive' in namespace and j == 0)):
                        ytitle = header[c].replace('-', '-\n') + '\n(PA-DA specific sequences)'
                else:
                    ytitle = header[c].replace('-', '-\n')
                a.set_ylabel(ytitle, color=color)
            if xticks:
                a.set_xticks(epochs)
                a.set_xticklabels(xticks)
            else:
                xticks_prim = [1] + [el for el in np.arange(0, len(epochs), math.ceil(max(epochs)/num_xticks))][1:]
                if 0 in epochs and len(epochs) - 1 not in xticks_prim:
                    xticks_prim.append(len(epochs) - 1)
                elif 0 not in epochs and len(epochs) not in xticks_prim:
                    xticks_prim.append(len(epochs))
                a.set_xticks(xticks_prim)
            if value.shape[-1] == len(neurons):  # check number of values for 1st epoch
                if args.plot_one:
                    y = [el[j] for el in value[j]]
                    plot_one(a, epochs, y, '.', neurons[j], colors[neurons[j]])
                else:
                    for k, n in enumerate(neurons):  # for each neuron
                        y = [el[k] for el in value[j]]
                        plot_one(a, epochs, y, '.', n, colors[neurons[k]])
            elif len(value[j][0]) == 1:  # or for single values
                plot_one(a, epochs, value[j], '.', 'general', COLORS[-1])
            if args.print_mean and len(value[j]) == len(neurons):
                y = [mean(el) for el in value[j]]
                plot_one(a, epochs, y, 'x', 'mean', COLORS[-2])
        a.set_yticklabels([str(el).rstrip('0').rstrip('.') if len(str(el)) < 4
                           else str(round(el, 4)).rstrip('0').rstrip('.') for el in a.get_yticks()])
        a.grid(True, color='white')
if namespace in ['basset3{}'.format(i) for i in range(0, 5)]:
    fig.suptitle('Basset {}'.format(int(namespace.split('3')[1]) + 1), fontsize=18)
elif namespace in ['custom4{}'.format(i) for i in range(0, 5)]:
    fig.suptitle('Custom {}'.format(int(namespace.split('4')[1]) + 1), fontsize=18)
elif 'positive' in namespace:
    if '1561' in namespace:
        network_number = 1
        num_seqs = 1561
    elif '6000' in namespace:
        network_number = 2
        num_seqs = 6000
    elif '10000' in namespace:
        network_number = 3
        num_seqs = 10000
    else:
        network_number = ''
        num_seqs = namespace.split('_')[-1]
    grade_group = namespace.split('-positive')[0].upper()
    if args.subset:
        fig.suptitle('{}-positive {}; specific subset (1561 out of {} sequences)'.
                     format(grade_group, network_number, num_seqs))
    else:
        fig.suptitle('{}-positive {}; {} sequences'.format(grade_group, network_number, num_seqs))

elif 'patient' in namespace and 'specific' in namespace:
    if '7842' in namespace:
        network_number = 1
        num_seqs = 7842
    elif '20000' in namespace:
        network_number = 2
        num_seqs = 20000
    elif '40000' in namespace:
        network_number = 3
        num_seqs = 40000
    else:
        network_number = ''
        num_seqs = namespace.split('_')[-1]
    if args.subset:
        fig.suptitle('Patient-specific {}; specific subset (7842 out of {} sequences)'.format(network_number, num_seqs))
    else:
        fig.suptitle('Patient-specific {}; {} sequences'.format(network_number, num_seqs))
else:
    fig.suptitle(namespace, fontsize=18)
plt.subplots_adjust(wspace=0.05)
handles, labels = [], []
for a in axes.flatten():
    h, l = a.get_legend_handles_labels()
    handles += h
    labels += l
if len(set(labels)) < len(labels):
    handles_unique, labels_unique = [], []
    for h, l in zip(handles, labels):
        if l not in labels_unique:
            handles_unique.append(h)
            labels_unique.append(l)
    handles = handles_unique
    labels = labels_unique
#fig.legend(handles, labels, loc='upper center')
if 'loss' in header[c].lower():
    axes[-1][0].legend(handles, labels, bbox_to_anchor=(1, -0.08), loc="upper center", ncol=4)
else:
    if args.subset or '7842' in namespace or '1561' in namespace:
        axes[-1][0].legend(handles, labels, bbox_to_anchor=(1, -0.17), loc="upper center", ncol=4)
    else:
        axes[-1][0].legend(handles, labels, bbox_to_anchor=(1, -0.37), loc="upper center", ncol=4)
#axes[-1][0].legend(bbox_to_anchor=(0, -0.07), loc="upper left", ncol=4)
plt.show()
plotname = '-'.join([s.lower().replace('_', '') for s in stages]) + ':' + '-'.join([el.lower() for el in columns])
fig.savefig(os.path.join(output, namespace + '_{}.png'.format(plotname)))
