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

try:
    values = np.nan_to_num(values)
    ylims = [np.min(values) - 0.05, np.max(values) + 0.05]
except ValueError:
    print('No values were read from the results file!')
    raise ValueError


def plot_one(ax, x, y, line, label, color):
    if not all([el == 0 for el in y]):
        ax.plot(x, y, line, label=label, alpha=0.5, color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylim(*ylims)


neurons = get_classes_names(param)

if cv:
    colnum = colnum[:1]
fig, axes = plt.subplots(nrows=len(colnum), ncols=len(stages), figsize=(12, 8), squeeze=False)
if axes.shape[1] > 1:
    num_xticks = 10
else:
    num_xticks = 20
for i, (stage, value) in enumerate(zip(stages, values)):  # for each stage
    if args.subset:
        title = STAGES[stage.replace('subset_', '')] + ' - subset'
    else:
        title = STAGES[stage]
    axes[0, i].set_title(title)
    for j, c in enumerate(colnum):  # for each column
        a = axes[j][i]
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
                        color = COLORS[neurons.index(n)]
                a.set_ylabel(header[c].replace('-', '-\n'), color=color)
            if xticks:
                a.set_xticks(epochs)
                a.set_xticklabels(xticks)
            else:
                a.set_xticks([el for el in np.arange(1, len(epochs), math.ceil(len(epochs)/num_xticks))] + [len(epochs)])
            if len(value[j][0]) == len(neurons):  # check number of values for 1st epoch
                if args.plot_one:
                    y = [el[j] for el in value[j]]
                    plot_one(a, epochs, y, '.', neurons[j], COLORS[j])
                else:
                    for k, n in enumerate(neurons):  # for each neuron
                        y = [el[k] for el in value[j]]
                        plot_one(a, epochs, y, '.', n, COLORS[k])
            elif len(value[j][0]) == 1:  # or for single values
                plot_one(a, epochs, value[j], '.', 'general', COLORS[-1])
            if args.print_mean and len(value[j]) == len(neurons):
                y = [mean(el) for el in value[j]]
                plot_one(a, epochs, y, 'x', 'mean', COLORS[-2])

fig.suptitle(namespace)
#axes[-1][0].legend(bbox_to_anchor=(0, -0.07*(i+j+1)), loc="upper left", ncol=4)
#axes[-1][0].legend(bbox_to_anchor=(0, -0.07), loc="upper left", ncol=4)
plt.show()
plotname = '-'.join([s.lower().replace('_', '') for s in stages]) + ':' + '-'.join([el.lower() for el in columns])
fig.savefig(os.path.join(output, namespace + '_{}.png'.format(plotname)))
