import argparse
from bin.common import *
from bin.datasets import SeqsDataset
import torch
from time import time
from bin.integrated_gradients import integrated_gradients
import warnings
from Bio import SeqIO

parser = argparse.ArgumentParser(description='Calculate integrated gradients based on given sequences and '
                                             'network')
parser.add_argument('seq', metavar='FILE', type=str,
                    help='File with sequences to check')
parser.add_argument('--model', action='store', metavar='NAME', type=str, default=None,
                    help='File with the model to check, if PATH is given, model is supposed to be in PATH directory, '
                         'if NAMESPACE is given model is supposed to be in [PATH]/results/[NAMESPACE]/ directory')
parser.add_argument('--baseline', action='store', metavar='DATA', type=str, default=None,
                    help='Baseline for calculating integrated gradients: None/fixed, random, zeros or npy/fasta file. '
                         'By default is None - random baseline, the same for all sequences, is used')
parser.add_argument('--trials', action='store', metavar='NUM', type=int, default=10,
                    help='Number of trials for calculating integrated gradients, default = 10.')
parser.add_argument('--steps', action='store', metavar='NUM', type=int, default=50,
                    help='Number of steps for each trial, default = 50.')
parser.add_argument('--all_classes', action='store_true',
                    help='Calculate gradients for all neurons (by default only output for the real label is calculated)')
parser.add_argument('--integrads_name', action='store', metavar='NAME', type=str, default=None,
                    help='Alternative name for the output directory')
parser = basic_params(parser, param=True)
args = parser.parse_args()

if args.namespace is None:
    namesp = os.path.dirname(args.seq).strip('/').split('/')[-1]
else:
    namesp = args.namespace
path, output, namespace, seed = parse_arguments(args, args.seq, namesp=namesp, model_path=True)

if args.model is None:
    model_file = os.path.join(path, '{}_last.model'.format(namespace, namespace))
elif os.path.isfile(args.model):
    model_file = args.model
else:
    model_file = os.path.join(path, args.model)
model_path = os.path.dirname(model_file)

if args.param is not None:
    if os.path.isfile(args.param):
        param_file = args.param
    elif os.path.isfile(os.path.join(path, args.param)):
        param_file = os.path.isfile(os.path.join(path, args.param))
    else:
        warnings.warn('Param file could not be found!')
elif args.param is None:
    param_dir = os.path.join(model_path, '{}_params.txt'.format(namespace))
    if os.path.isfile(param_dir):
        param_file = param_dir
    else:
        warnings.warn('Param file cannot be found in location: {}'.format(param_dir))

if os.path.isfile(args.seq):
    seq_file = args.seq
elif os.path.isfile(os.path.join(model_path, args.seq)):
    seq_file = os.path.join(model_path, args.seq)
else:
    warnings.warn('Neither {} nor {} does not exist!'.format(args.seq, os.path.join(model_path, args.seq)))
_, seq_name = os.path.split(seq_file)
seq_name, _ = os.path.splitext(seq_name)

# CUDA for PyTorch
use_cuda, device = check_cuda(None)

network, _, seq_len, _, classes, analysis_name, num_epochs = params_from_file(param_file)
trials = args.trials

dataset = SeqsDataset(seq_file, seq_len=seq_len, name_pos=args.name_pos)
assert classes == dataset.classes, 'List of classes is inconsistent'
seq_ids = dataset.IDs
seq_desc = []
for i in seq_ids:
    seq_desc.append(dataset.__getitem__(i, info=True)[7])
X, y = dataset.__getitem__(0)
labels = [y]
X = [X]
for i in range(1, len(dataset)):
    xx, yy = dataset.__getitem__(i)
    X.append(xx)
    labels.append(yy)
X = torch.stack(X, dim=0)
num_seqs_query = X.shape[0]


save_baseline = False
if args.baseline is None or args.baseline == 'fixed':
    from bin.common import OHEncoder
    import random
    encoder = OHEncoder()
    base = []
    for _ in range(trials):
        b = np.zeros(X.shape)
        for j, el in enumerate(X):
            seq = random.choices(encoder.dictionary, k=el.shape[-1])
            b[j] = encoder(seq)
        base.append(b)
    base = np.stack(base)
    baseline_file = '{}_baseline.npy'.format(seq_name.replace('_', '-'))
    baseline_name = 'fixed'
    save_baseline = True
elif args.baseline == 'random':
    base = None
    baseline_mode = baseline_name = 'random'
    print('Baseline set to random - different for each sequence')
elif args.baseline == 'zeros':
    base = np.stack([(0 * X).numpy() for _ in range(trials)])
    baseline_mode = baseline_name = 'zeros'
    print('Baseline set to zero array')
else:
    baseline_mode = args.baseline
    _, baseline_name = os.path.split(baseline_mode)
    baseline_name, _ = os.path.splitext(baseline_name)
    if args.baseline.endswith('npy'):
        base = np.load(args.baseline, allow_pickle=True)
        assert base.shape[1] == num_seqs_query, 'Baseline shape: {}, Seqs shape: {}'.format(base.shape, X.shape)
        print('Baseline loaded from {}, size: {}'.format(args.baseline, base.shape))
    elif args.baseline.endswith('fasta'):
        base_seqs = []
        for record in SeqIO.parse(args.baseline, "fasta"):
            base_seqs.append(str(record.seq))
        encoder = OHEncoder()
        b = np.stack([np.array(encoder(seq.upper())) for seq in base_seqs], axis=0)
        base = np.stack([b for _ in range(num_seqs_query)], axis=1)
        n = len(base_seqs)
        baseline_name = '{}_{}-{}_baseline'.format(baseline_name, num_seqs_query, n)
        baseline_file = baseline_name + '.npy'
        save_baseline = True
        print('Baseline created from {}, size: {}'.format(args.baseline, base.shape))
    else:
        raise Exception('Unknown baseline mode: {}'.format(args.baseline))
    trials = base.shape[0]

if args.integrads_name is not None:
    integrads_name = args.integrads_name
else:
    integrads_name = 'integrads_{}_{}_{}_{}-{}'.format(analysis_name,
                                                       seq_name.replace('_', '-'),
                                                       baseline_name.replace('_', '-'),
                                                       trials,
                                                       args.steps)
outdir = os.path.join(output, integrads_name)
if os.path.isdir(outdir):
    warnings.warn('\nAnalysis in {} already exists, it will be overwritten'.format(outdir))
    import shutil
    shutil.rmtree(outdir)
os.mkdir(outdir)

if save_baseline:
    baseline_file = os.path.join(outdir, baseline_file)
    np.save(baseline_file, base)
    print('Baseline was written into {}'.format(baseline_file))
    baseline_mode = baseline_file

t0 = time()
# Build network
model = network(seq_len)
# Load weights from the file
model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
print('Model from {} loaded in {:.2f} s'.format(model_file, time() - t0))

analysis_info = os.path.join(outdir, 'params.txt')
with open(analysis_info, 'w') as f:
    f.write('Model file: {}\n'.format(model_file))
    f.write('Seq file: {}\n'.format(seq_file))
    f.write('Seq IDs: {}\n'.format(', '.join(seq_ids)))
    f.write('Seq labels: {}\n'.format(', '.join(list(map(str, map(int, labels))))))
    f.write('Seq length: {}\n'.format(seq_len))
    f.write('Seq descriptions: {}\n'.format(', '.join(seq_desc)))
    f.write('Classes: {}\n'.format(', '.join(classes)))
    f.write('Number of trials: {}\n'.format(trials))
    f.write('Number of steps: {}\n'.format(args.steps))
    f.write('Baseline: {}\n'.format(baseline_mode))
print('Analysis info written into {}'.format(analysis_info))

leap = 100
t0 = time()
if args.all_classes:
    results = {}
    for i, name in enumerate(classes):
        l = [i for _ in labels]
        print('Calculating integrated gradients for {}'.format(name))
        r = np.squeeze(
            integrated_gradients(model, X, l, use_cuda=use_cuda, num_trials=trials, steps=args.steps,
                                 baseline=base), axis=1)
        np.save(os.path.join(outdir, 'integrads_{}'.format('-'.join(name.split()))), r)
        results[name] = r
        print('---> Total elapsed time: {:.2f} min'.format((time() - t0) / 60))
else:
    print('Calculating integrated gradients for true class')
    r = np.squeeze(integrated_gradients(model, X, labels, use_cuda=use_cuda, num_trials=trials, steps=args.steps,
                                        baseline=base), axis=1)
    np.save(os.path.join(outdir, 'integrads_all'), r)
    print('---> Total elapsed time: {:.2f} min'.format((time() - t0) / 60))
print('Gradients calculated in {:.2f} min and saved into {} directory'.format((time() - t0) / 60, outdir))
