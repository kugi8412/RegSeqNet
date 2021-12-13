import os
import argparse
from bin.common import *
from natsort import natsorted

parser = argparse.ArgumentParser(description='Create set of sequences of given length based on whole hg38 genome')
parser.add_argument('hg38_data', action='store', metavar='DATA', type=str,
                    help='Folder with the hg38 genome, if PATH is given, data is supposed to be ' +
                         'in PATH directory: [PATH]/data/[DATA]')
parser.add_argument('--seq_length', action='store', metavar='INT', type=int, default=2000,
                    help='Length of output sequences, default is 2000')
parser.add_argument('--interval', action='store', metavar='INT', type=int, default=100,
                    help='Interval between successive sequences, default is 100')
parser.add_argument('--batch_size', action='store', metavar='INT', type=int, default=64,
                    help='size of the batch, default: 64')
parser.add_argument('--constant_class', action='store', metavar='CLASS', type=str, default='promoter active',
                    help='Class that should be assigned to all created sequences, default is promoter active')
parser = basic_params(parser)
args = parser.parse_args()
path, output, namespace, seed = parse_arguments(args, args.hg38_data)

hg38_path, seq_length, interval, batch_size, constant_class = args.hg38_data, args.seq_length, args.interval, \
                                                              args.batch_size, args.constant_class

chr_files = [el for el in os.listdir(os.path.join(path, 'data', hg38_path))
             if el.startswith('chr') and el.endswith('.fasta')]

class_mapping = {
    'promoter active': 1,
    'nonpromoter active': 2,
    'promoter inactive': 3,
    'nonpromoter inactive': 4
}

if not os.path.exists(output):
    os.mkdir(output)
    print('Output directory "{}" created.'.format(output))

print(output)

for file in natsorted(chr_files):
    chr_number = file.replace('chr', '').replace('.fasta', '')
    print('Reading sequence of chr {}'.format(chr_number))
    output_file = os.path.join(output, file.replace('.fasta', '_{}_{}_class{}.fasta'.format(seq_length, interval,
                                                                                            class_mapping[constant_class])))
    output_writer = open(output_file, 'w')
    file_path = os.path.join(path, 'data', hg38_path, file)
    with open(file_path, 'r') as f:
        line = f.readline()
        while line.startswith('>'):
            line = f.readline()
        seq = line.strip()
    start, end = 0, seq_length
    i = 0
    while end < len(seq):
        header = '> chr{} {}-{} + {}'.format(chr_number, start+1, end, constant_class)
        clip = seq[start:end]
        output_writer.write('{}\n{}\n'.format(header, clip))
        start += interval
        end += interval
        i += 1
    output_writer.close()
    print('Chr {}: {} sequences saved to "{}"'.format(chr_number, i, output_file))
