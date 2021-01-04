import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from bin.common import *

COLORS = ['C{}'.format(i) for i in range(10)]

parser = argparse.ArgumentParser(description='Compare network outputs for reference and for alternative sequences')
parser.add_argument('plot_type', choices=['scatter', 'boxplot', 'barplot'], metavar='TYPE',
                    help='Type of the plot, choose: scatter, boxplot, barplot')
parser.add_argument('--name', '--test_namespace', metavar='NAMES', nargs='+', default=['test'],
                    help='Namespaces of test analyses, default: test')
parser.add_argument('--name_pos', action='store', metavar='INT', nargs='+', default=None,
                    help='Position(s) of sequence name in the fasta header, by default created as CHR:POSITION')
parser = basic_params(parser)
args = parser.parse_args()
path, outdir, namespace, seed = parse_arguments(args, None, model_path=True)


def load_data(name):
    name = name.replace('_', '-')
    outputs_file = os.path.join(path, '{}_{}_outputs.npy'.format(namespace, name))
    outputs = np.load(outputs_file, allow_pickle=True)
    print('Loaded network outputs from {}'.format(outputs_file))
    labels_file = os.path.join(path, '{}_{}_labels.npy'.format(namespace, name))
    labels = list(np.load(labels_file, allow_pickle=True))
    print('Loaded sequences labels from {}'.format(labels_file))
    ids_file = os.path.join(path, '{}_{}.txt'.format(namespace, name))
    seq_ids = open(ids_file, 'r').read().strip().split('\n')
    print('Loaded sequences IDs from {}'.format(ids_file))
    num_seqs = len(seq_ids)
    seq_file = None
    with open(os.path.join(path, '{}_test_results.tsv'.format(namespace)), 'r') as f:
        f.readline()
        for line in f:
            line = line.strip().split('\t')
            if line[1] == name:
                seq_file = line[0]
                break
    return outputs, labels, seq_ids, num_seqs, seq_file, name


def plot_scatter(name):
    outputs, labels, seq_ids, num_seqs, seq_file, name = load_data(name)

    label_names = ['' for _ in range(num_seqs)]
    if os.path.isfile(seq_file):
        seqs = ['' for _ in range(num_seqs)]
        patients = ['' for _ in range(num_seqs)]
        ref_seq = None
        with open(seq_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    l = line.strip('>\n ').split(' ')
                    if args.name_pos is not None:
                        name_pos = [int(el) for el in args.name_pos]
                        id = '-'.join(list(np.array(l)[name_pos]))
                    else:
                        id = '{}_{}'.format(l[0].lstrip('chr'), l[1])
                    pos = seq_ids.index(id)
                    label_names[pos] = '{} {}'.format(l[3], l[4])
                    patients[pos] = id
                else:
                    if l[-1] == 'REF':
                        ref_seq = line.strip().upper()
                    seqs[pos] = line.strip().upper()
        if ref_seq is not None:
            num_snp = [len([a for a, r in zip(seq, ref_seq) if a != r]) for seq in seqs]
            min_nsnp = min([el for el in num_snp if el != 0])
            max_nsnp = max([el for el in num_snp if el != 0])
            dots = [(el - min_nsnp + 1) * 30 if el != 0 else 12 for el in num_snp]
        else:
            min_nsnp, max_nsnp = 0, 0
            dots = [12 for _ in seqs]

    else:
        patients = seq_ids
        dots = [12 for _ in range(len(seq_ids))]
    print('Alternative and reference sequences read from {}'.format(seq_file))

    classes = get_classes_names(os.path.join(path, '{}_params.txt'.format(namespace)))
    xvalues = {'True class': [], 'False class': []}
    yvalues = {'True class': [], 'False class': []}
    sizes = {'True class': [], 'False class': []}

    correct_classified = 0
    for i, (label, n) in enumerate(zip(labels, label_names)):
        output = outputs[label]
        seq_pos = len([el for el in labels[:i] if el == label])
        xvalues['True class'].append(label * num_seqs + i + label + 1)
        correct_out = output[label][seq_pos]
        yvalues['True class'].append(correct_out)
        sizes['True class'].append(dots[i])
        classified = True
        for wrong_name in [el for el in classes if el != n]:
            wrong_label = classes.index(wrong_name)
            xvalues['False class'].append(wrong_label * num_seqs + i + wrong_label + 1)
            wrong_out = output[wrong_label][seq_pos]
            yvalues['False class'].append(wrong_out)
            if wrong_out >= correct_out:
                classified = False
            sizes['False class'].append(dots[i])
        if classified:
            correct_classified += 1
    print('Number of sequences: {}, number of classes: {}'.format(num_seqs, len(classes)))

    plt.figure(figsize=(20, 10))
    for legend_label, color, marker in zip(['True class', 'False class'], ['C2', 'C1'], ['*', 'o']):
        plt.scatter(xvalues[legend_label], yvalues[legend_label], s=sizes[legend_label], color=color, marker=marker,
                    label=legend_label, alpha=0.8)
    xticks = [la for el in xvalues.values() for la in el]
    xticks.sort()
    plt.xticks(xticks, patients*len(classes), fontsize=10, rotation=90, ha='center')
    plt.xlabel(('  ' * num_seqs).join(classes), fontsize=16)
    plt.ylabel('Output value', fontsize=16)
    plt.legend(fontsize=12, prop={'size': 16})
    plt.title('{} - {}'.format(namespace, name), fontsize=20)
    plt.ylim((0.45, 1.05))
    if min_nsnp == max_nsnp == 0:
        plt.text(0.0, 1.045, 'Correctly classified seqs: {}/{}'.
                 format(correct_classified, num_seqs), fontsize=12, va='top')
    else:
        plt.text(0.0, 1.045, 'Correctly classified seqs: {}/{}\nNumber of SNPs in alt seqs: {}-{}'.
                 format(correct_classified, num_seqs, min_nsnp, max_nsnp), fontsize=12, va='top')
    plt.tight_layout()
    plot_file = os.path.join(outdir, '{}_{}_ref:alt.png'.format(namespace, name))
    plt.savefig(plot_file)
    plt.show()
    print('Plot saved to {}'.format(plot_file))


def get_outputs_and_snps(name, accuracy=False):
    outputs, labels, seq_ids, num_seqs, seq_file, name = load_data(name)
    label_names = ['' for _ in range(num_seqs)]
    patients = ['' for _ in range(num_seqs)]
    snps = [0 for _ in range(num_seqs)]
    with open(seq_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                l = line.strip('>\n ').split(' ')
                if args.name_pos is not None:
                    name_pos = [int(el) for el in args.name_pos]
                    id = '-'.join(list(np.array(l)[name_pos]))
                else:
                    id = '{}_{}'.format(l[0].lstrip('chr'), l[1])
                try:
                    pos = seq_ids.index(id)
                except ValueError:
                    print('WARNING: sequence {} omitted'.format(id))
                    continue
                label_names[pos] = '{} {}'.format(l[3], l[4])
                patients[pos] = id
                if l[6] != 'REF':
                    snps[pos] = int(l[7].rstrip('SNPs'))
    if not accuracy:
        output_data, num_snps = [], []
        for i, label in enumerate(labels):
            output = outputs[label]
            seq_pos = len([el for el in labels[:i] if el == label])
            correct_out = output[label][seq_pos]
            output_data.append(correct_out)
        return output_data, snps
    else:
        accuracy = []
        for i, label in enumerate(labels):
            output = outputs[label]
            seq_pos = len([el for el in labels[:i] if el == label])
            if output[label][seq_pos] == max([el[seq_pos] for el in output]):
                accuracy.append(1)
            else:
                accuracy.append(0)
        return accuracy, snps


def plot_boxplot(name):
    output_data, num_snps = get_outputs_and_snps(name)
    ref_output, ref_snps = get_outputs_and_snps(name.replace('-pos', '-ref').replace('-neg', '-ref'))
    output_data += ref_output
    num_snps += ref_snps
    num_boxes = 7
    box_size = max(num_snps) // (num_boxes - 1)
    nsnp_range = [[0, 0], [1, box_size]] + [[i*box_size+1, (i+1)*box_size] for i in range(1, num_boxes-2)] + \
                 [[(num_boxes-1)*box_size, np.inf]]
    y_values = [[el for el, la in zip(output_data, num_snps) if v1 <= la <= v2]
                for (v1, v2) in nsnp_range]
    x_ticks = [str(el[0]) if el[1] == el[0] else '>={}'.format(el[0]) if el[1] == np.inf
               else '{}-{}'.format(el[0], el[1]) for el in nsnp_range]
    print(len(num_snps))
    #plt.hist(num_snps, bins=[0, 1, 2, 3, 4, 5, 50, 100])
    plt.boxplot(y_values)
    plt.xticks(range(1, len(x_ticks)+1), x_ticks)
    plt.ylabel('Output from a true neuron')
    if '-pos' in name:
        plt.title('Patient-specific: reference + positive examples')
    elif '-neg' in name:
        plt.title('Patient-specific: reference + negative examples')
    plt.show()


def plot_barplot(name, num_boxes=None, ref=False):
    from statistics import mean
    accuracy, num_snps = get_outputs_and_snps(name, accuracy=True)
    if ref:
        ref_accuracy, ref_snps = get_outputs_and_snps(name.replace('-pos', '-ref').replace('-neg', '-ref'))
        accuracy += ref_accuracy
        num_snps += ref_snps
        nsnp_range = [[0, 0]]
    else:
        nsnp_range = []
    if num_boxes is None:
        nsnp_range += [[1, 1], [2, 2], [3, 3], [4, 4], [5, 10], [11, np.inf]]
        n_seqs = [len([el for el in num_snps if v1 <= el <= v2]) for (v1, v2) in nsnp_range]
        widths = [el*0.001 for el in n_seqs]
        widths_prim = [sum(widths[:i]) if i > 0 else 0 for i in range(len(widths))]
        gap_width = 0.1 * widths[0]
        x_values = [el+la/2+i*gap_width for i, (el, la) in enumerate(zip(widths_prim, widths))]
    else:
        box_size = max(num_snps) // (num_boxes - 1)
        nsnp_range += [[1, box_size]] + [[i * box_size + 1, (i + 1) * box_size] for i in range(1, num_boxes - 2)] + \
                      [[(num_boxes - 1) * box_size, np.inf]]
        x_values = range(1, len(nsnp_range) + 1)
        widths = 0.8
    y_values = [mean(selected) if len(selected) > 0 else 0 for selected in
                [[el for el, la in zip(accuracy, num_snps) if v1 <= la <= v2] for (v1, v2) in nsnp_range]]
    x_ticks = [str(el[0]) if el[1] == el[0] else '>={}'.format(el[0]) if el[1] == np.inf
               else '{}-{}'.format(el[0], el[1]) for el in nsnp_range]
    plt.bar(x_values, y_values, width=widths)
    if num_boxes is None:
        for x, y, text in zip(x_values, y_values, n_seqs):
            plt.text(x, y+0.01, text, ha='center')
    plt.xticks(x_values, x_ticks, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of SNPs')
    if '-pos' in name:
        title = 'Patient-specific: positive examples'
    elif '-neg' in name:
        title = 'Patient-specific: negative examples'
    if ref:
        title += ' + reference'
    plt.title(title)
    plt.show()


for name in args.name:
    print('\nPlot "{}" for {}'.format(args.plot_type, name))
    if args.plot_type == 'scatter':
        plot_scatter(name)
    elif args.plot_type == 'boxplot':
        plot_boxplot(name)
    elif args.plot_type == 'barplot':
        plot_barplot(name)
    else:
        raise ValueError
