import matplotlib.pyplot as plt
import argparse
from bin.common import *
from statistics import median

COLORS = ['C{}'.format(i) for i in range(10)]

parser = argparse.ArgumentParser(description='Plot integrated gradients for given namespace '
                                             '(before it run calculate_integrads in order to create required files '
                                             'with gradients).')
parser = basic_params(parser)
parser.add_argument('--all_classes', action='store_true',
                    help='Plot outputs for all neurons (by default only output for the real label is showed)')
parser.add_argument('--single', action='store_true',
                    help='Plot single scatter plot for each sequence')
parser.add_argument('--clip', action='store', metavar='NUMBER', type=int, default=None,
                    help='Number of +- subset of nucleotides from the middle of the sequence to plot')
args = parser.parse_args()
path, output, namespace, seed = parse_arguments(args, None, model_path=True)

param_file = os.path.join(path, 'params.txt')
with open(param_file) as f:
    for line in f:
        if line.startswith('Model file'):
            _, analysis_name = os.path.split(line.split(': ')[1].strip())
            analysis_name = analysis_name.split('_')[0]
        elif line.startswith('Seq file'):
            seq_file = line.split(': ')[1].strip()
        elif line.startswith('Seq IDs'):
            seq_ids = line.split(': ')[1].strip().split(', ')
        elif line.startswith('Seq length'):
            seq_len = int(line.split(': ')[1].strip())
        elif line.startswith('Seq labels'):
            seq_labels = list(map(int, line.split(': ')[1].strip().split(', ')))
        elif line.startswith('Seq descriptions'):
            seq_desc = line.split(': ')[1].strip().split(', ')
        elif line.startswith('Classes'):
            classes = line.split(': ')[1].strip().split(', ')
        elif line.startswith('Number of trials'):
            trials = int(line.split(': ')[1].strip())
        elif line.startswith('Number of steps'):
            steps = int(line.split(': ')[1].strip())

if 'seq_desc' in globals():
    seq_names = ['{}\n{}'.format(el, la) for el, la in zip(seq_ids, seq_desc)]
else:
    seq_names = seq_ids

order = [0 for _ in seq_names]
for i, (name, label) in enumerate(zip(seq_desc, seq_labels)):
    w = 0 if 'best' in name else 1
    order[2*label + w] = i

seq_names = list(np.array(seq_names)[order])
seq_labels = list(np.array(seq_labels)[order])

try:
    results = {}
    for name in classes:
        results[name] = np.load(os.path.join(path,
                                             'integrads_{}_{}.npy'.format(namespace, '-'.join(name.split()))))[order]
except FileNotFoundError:
    results = np.load(os.path.join(path, 'integrads_all.npy'))[order]

leap = 40
if args.single:
    min_value, max_value = 0, 0
    tt = 4
    for n, (seq, label) in enumerate(zip(seq_names, seq_labels)):
        if args.all_classes:
            fig, axes = plt.subplots(nrows=4, ncols=len(classes), figsize=(16, 10), squeeze=False, sharex='col',
                                     sharey='row', gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
            arg_classes = classes
        else:
            fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(16, 10), squeeze=False, sharex='col',
                                     sharey='row', gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
            arg_classes = ['True label']
        for i, name in enumerate(arg_classes):
            for j, letter in enumerate(['A', 'C', 'G', 'T']):
                ax = axes[j, i]
                if j == 0:
                    if args.all_classes:
                        cc = COLORS[i]
                    else:
                        cc = 'black'
                    ax.set_title(name, fontsize=15, color=cc)
                if i == 0:
                    ax.set_ylabel(letter, fontsize=15, rotation='horizontal', ha='right', va='center')
                if args.all_classes:
                    result = results[name][n][j]
                elif isinstance(results, dict):
                    result = results[classes[seq_labels[n]]][n][j]
                else:
                    result = results[n][j]
                if args.clip is not None:
                    start_point = int(seq_len/2 - args.clip)
                    result = [median(result[start_point+i : start_point+i+leap]) for i in range(0, 2*args.clip, leap)]
                    new_len = 2*args.clip
                    xpos = np.arange(leap / 2, new_len + 0.5, leap)
                    ax.scatter(xpos, result, marker=".", color=COLORS[label])
                    xticks = [i for i in np.arange(0, new_len + 0.5, new_len // 4)]
                    labels = [str(int(i)) for i in np.arange(start_point, start_point + new_len + 0.5, new_len // 4)]
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(labels)
                else:
                    result = [median(result[i:i + leap]) for i in range(0, seq_len, leap)]
                    xpos = np.arange(leap/2, seq_len + 0.5, leap)
                    ax.scatter(xpos, result, marker=".", s=(72./fig.dpi)*20*(math.sqrt(leap)), edgecolor="None", color=COLORS[label])
                    xticks = [i for i in np.arange(0, seq_len + 0.5, seq_len // 4)]
                    ax.set_xticks(xticks)
                if n == 0:
                    min_value = [tt*np.min(result) if tt*np.min(result) < min_value else min_value][0]
                    max_value = [tt*np.max(result) if tt*np.max(result) > max_value else max_value][0]
        for ax in axes.flatten():
            ax.set_ylim((-0.01, 0.01))

        if args.clip:
            fig.suptitle('Integrated gradients - seq {}; {}; clipped +- {}'.format(seq.replace('\n', '; '), classes[label], args.clip),
                         color=COLORS[label], fontsize=15)
            plt.tight_layout()
            plt.show()
            fig.savefig(os.path.join(output, 'integrads_{}_{}_clip{}_seq{}.png'.format(namespace, leap, args.clip, n)))
        else:
            fig.suptitle(
                'Integrated gradients - seq {}; {}'.format(seq.replace('\n', '; '), classes[label]),
                color=COLORS[label], fontsize=15)
            plt.tight_layout()
            plt.show()
            fig.savefig(os.path.join(output, 'integrads_{}_{}_seq{}.png'.format(namespace, leap, n)))
else:
    if args.all_classes:
        fig, axes = plt.subplots(nrows=len(seq_names), ncols=len(classes), figsize=(12, 8), squeeze=False, sharex='col',
                                 sharey='row', gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
        arg_classes = classes
    else:
        fig, axes = plt.subplots(nrows=len(seq_names), ncols=1, figsize=(12, 8), squeeze=False, sharex='col',
                                 sharey='row', gridspec_kw={'hspace': 0.12, 'wspace': 0.05})
        arg_classes = ['True label']
    min_value, max_value = 0, 0
    for i, name in enumerate(arg_classes):
        for j, seq in enumerate(seq_names):
            ax = axes[j, i]
            if j == 0:
                if args.all_classes:
                    cc = COLORS[i]
                else:
                    cc = 'black'
                    for cl in classes:
                        axes[-1, -1].plot([], label=cl)
                #ax.set_title(name, fontsize=15, color=cc)
            elif j == len(seq_names)-1:
                ax.set_xlabel('Sequence position')
            if i == 0:
                ax.set_ylabel(seq, fontsize=8, color=COLORS[seq_labels[j]], rotation='horizontal', ha='right', va='center')
            if args.all_classes:
                result = results[name][j]
            elif isinstance(results, dict):
                result = results[classes[seq_labels[j]]][j]
            else:
                result = results[j]
            if args.clip is not None:
                start_point = int(seq_len / 2 - args.clip)
                result = [result[:, start_point + i:start_point + i + leap].flatten() for i in
                          range(0, 2 * args.clip, leap)]
                new_len = 2 * args.clip
                xticks = [i for i in np.arange(0, new_len + 0.5, new_len // 4)]
                labels = [str(int(i)) for i in np.arange(start_point, start_point + new_len + 0.5, new_len // 4)]
            else:
                result = [result[:, i:i+leap].flatten() for i in range(0, seq_len, leap)]
                labels = [str(int(i)) for i in np.arange(0, seq_len + 0.5, seq_len // 4)]
                xticks = [int(el)//leap for el in labels]
            ax.boxplot(result, showfliers=True, whis=15.0)
            ax.set_xticks(xticks)
            ax.set_xticklabels(labels)
            min_value = [np.min(result) if np.min(result) < min_value else min_value][0]
            max_value = [np.max(result) if np.max(result) > max_value else max_value][0]
    for ax in axes.flatten():
        ax.set_ylim((min_value, max_value))
    if not args.all_classes:
        axes[-1, -1].legend(bbox_to_anchor=(0.08, -1.1), loc="lower left", ncol=4)

    #fig.suptitle('Integrated gradients - {}; {}; trials {}; steps {};'.format(*namespace.split('_')[:2], trials, steps),
    #             fontsize=15)
    fig.suptitle('Integrated gradients - Custom 1, zero baseline', fontsize=15)
    #plt.tight_layout()
    #plt.show()
    figpath = os.path.join(output, 'integrads_{}_{}.png'.format(namespace, leap))
    print('Plot saved to {}'.format(figpath))
    fig.savefig(figpath)
