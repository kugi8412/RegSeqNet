import argparse
from bin.common import *
from bin.datasets import *
import torch
import warnings
from datetime import datetime
from time import time
from statistics import mean
from natsort import natsorted

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Test given model on whole hg38 genome')
parser.add_argument('hg38_data', action='store', metavar='DATA', type=str,
                    help='Folder with the hg38 genome, if PATH is given, data is supposed to be ' +
                         'in PATH directory: [PATH]/data/[DATA]')
parser.add_argument('--model', action='store', metavar='NAME', type=str, default=None,
                    help='File with the model to test, if PATH is given, model is supposed to be in PATH directory, '
                         'if NAMESPACE is given model is supposed to be in [PATH]/results/[NAMESPACE]/ directory')
parser.add_argument('--interval', action='store', metavar='INT', type=int, default=100,
                    help='Interval between successive sequences, default is 100')
parser.add_argument('--batch_size', action='store', metavar='INT', type=int, default=64,
                    help='size of the batch, default: 64')
parser.add_argument('--chr_number', action='store', metavar='INT', type=int, default=22,
                    help='Number of chromosome to check.')
parser = basic_params(parser)
args = parser.parse_args()
path, output, namespace, seed = parse_arguments(args, args.hg38_data)

hg38_path, interval, batch_size, chr_number = \
    args.hg38_data, args.interval, args.batch_size, args.chr_number

chr_file = [el for el in os.listdir(os.path.join(path, 'data', hg38_path))
            if el.startswith('chr{}'.format(chr_number)) and el.endswith('.fasta')]
assert len(chr_file) == 1, chr_file
chr_file = chr_file[0]

if args.model is not None and os.path.isfile(args.model):
    modelfile = args.model
elif args.model is not None and os.path.isfile(os.path.join(path, 'results', namespace, args.model)):
    modelfile = os.path.join(path, namespace, args.model)
else:
    modelfile = os.path.join(path, 'results', namespace, '{}_last.model'.format(namespace))

if not os.path.exists(output):
    os.mkdir(output)

# Define loggers for logfile and for results
[logger, results_table], old_results = build_loggers('test', output=output, namespace='hg38')

logger.info('\nTesting the network {} begins {}\nInput data: chromosome {} from hg38 reference genome'
            '\nOutput directory: {}\n'.format(modelfile, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), chr_number, output))

# CUDA for PyTorch
use_cuda, device = check_cuda(logger)

network, _, seq_len, _, classes, _, _ = \
    params_from_file(os.path.join(path, 'results', namespace, '{}_params.txt'.format(namespace)))
class_mapping = {el: num for num, el in enumerate(classes)}
num_classes = len(classes)


def get_sequences(chr_number, chr_file, seq_len, interval, batch_size, logger):
    logger.info('Reading sequence of chr {}'.format(chr_number))
    file_path = os.path.join(path, 'data', hg38_path, chr_file)
    with open(file_path, 'r') as f:
        line = f.readline()
        while line.startswith('>'):
            line = f.readline()
        seq = line.strip()
    start, end = 0, seq_len
    while end < len(seq):
        seqs, names, j = [], [], 0
        num_random, num_omitted = 0, 0
        while j < batch_size:
            clip = seq[start:end]
            omit = False
            if 'N' in clip:
                pos = [i for i, el in enumerate(clip) if el == 'N']
                if len(pos) <= 0.05 * len(clip):
                    num_random += 1
                    clip = list(clip)
                    for p in pos:
                        clip[p] = random.choice(np.array(['A', 'C', 'G', 'T']))
                    clip = ''.join(clip)
                elif len(pos) == len(clip):
                    start += math.floor(seq_len / interval) * interval
                    end = start + seq_len
                    omit = True
                    num_omitted += math.floor(seq_len / interval)
                else:
                    start += interval
                    end += interval
                    omit = True
                    num_omitted += 1
            if not omit:
                seqs.append(str(clip).upper())
                j += 1
                names.append('> chr{} {} {} + promoter active'.format(chr_number, start, end))
                start += interval
                end += interval
            if end > len(seq):
                break
        if num_random > 0:
            logger.info('{} sequences contain <= 5% of unknown nucleotides - these nucleotides were changed to random one(s)'.
                        format(num_random))
        if num_omitted > 0:
            logger.info('{} sequences contain > 5% of unknown nucleotides - these sequences were omitted'.
                        format(num_omitted))
        yield seqs, names


def FakeLoader(seqs, names, batch_size, encoder):
    X_list, y_list = [], []
    i = 0
    for num, (seq, name) in enumerate(zip(seqs, names)):
        label = [class_mapping['{} {}'.format(*name.split(' ')[5:7])]]
        encoded_seq = encoder(seq)
        if encoded_seq is None:
            raise Exception('Sth went wrong - too many N in loader sequence')
        X = torch.tensor(encoded_seq)
        X = X.reshape(1, *X.size())
        X_list.append(X)
        y = torch.tensor(label)
        y_list.append(y)
        i += 1
        if i == batch_size or num == len(seqs) - 1:
            X = torch.stack(X_list, dim=0)
            y = torch.stack(y_list, dim=0)
            i = 0
            yield X, y


t0 = time()
# Build network - this type which was used during training the model
model = network(seq_len)
# Load weights from the file
model.load_state_dict(torch.load(modelfile, map_location=device))
if use_cuda:
    model.cuda()
logger.info('\nModel from {} loaded in {:.2f} s'.format(modelfile, time() - t0))

output_bed_files_names = ['output_chr{}_{}.bed'.format(chr_number, el.replace(' ', '_')) for el in classes]
output_bed_files = [open(os.path.join(output, el), 'w') for el in output_bed_files_names]

file_path = os.path.join(path, 'data', hg38_path, chr_file)
with open(file_path, 'r') as f:
    line = f.readline()
    while line.startswith('>'):
        line = f.readline()
    seq = line.strip()
chr_length = len(seq)
num_seqs = math.ceil((chr_length - seq_len) / interval) + 1
logger.info('Number of sequencess to test: {}'.format(num_seqs))
num_batches = math.ceil(num_seqs/batch_size)
logger.info('Number of batches: {}'.format(num_batches))

i = 0
logger.info('\n--- TESTING ---')
t0 = time()
for seqs, names in get_sequences(chr_number, chr_file, seq_len, interval, batch_size, logger):
    output_values = [[[] for _ in range(num_classes)] for _ in range(num_classes)]
    _, _, _, _, output_values, _, _ = \
        validate(model, FakeLoader(seqs, names, batch_size, OHEncoder()), num_classes, 1, use_cuda,
                 output_values=output_values, test=True)
    output_values = np.array(output_values[0])
    for num_class, file in enumerate(output_bed_files):
        for name, num_seq in zip(names, range(batch_size)):
            start, end = [int(el) for el in name.split(' ')[2:4]]
            file.write('chr{}\t{}\t{}\t{:.3f}\n'.format(chr_number, start, end, output_values[num_class, num_seq]))
    i += 1
    if i % 10 == 0:
        logger.info('Batch {}/{}'.format(i, num_batches))

logger.info("\nTesting of chromosome {} from hg38 based on {} finished in {:.2f} min\nResults saved to: {}"
            .format(chr_number, namespace, (time() - t0)/60, '\n'.join(output_bed_files_names)))
