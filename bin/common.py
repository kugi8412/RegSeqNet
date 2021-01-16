import os
from bin.networks import *
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder as Encoder
import numpy as np
import random

NET_TYPES = {
    'basset': BassetNetwork,
    'custom': CustomNetwork,
    'test': TestNetwork,
    'random-forest': 'RandomForest'
}

PARAMS = OrderedDict({
    'Name of the analysis': 'namespace',
    'Network type': 'network_name',
    'Network params': 'network_params',
    'Possible classes': 'classes',
    'Number of epochs': 'num_epochs',
    'Number of seqs': 'num_seqs',
    'Batch size': 'batch_size',
    'Training seqs': 'train_num',
    'Validation seqs': 'valid_num',
    'Test seqs': 'test_num',
    'Data directory': 'data_dir',
    'Random seed': 'seed',
    'CUDA available': 'use_cuda',
    'Optimizer': 'optimizer_name',
    'Loss function': 'lossfn_name',
    'Learning rate': 'lr',
    'Adjusting lr': 'adjust_lr',
    'Weight decay': 'weight_decay',
    'Retraining': 'modelfile'
})

SHORTCUTS = {
    'loss': 'Loss',
    'sens': 'Sensitivity',
    'spec': 'Specificity',
    'auc': 'AUC'
}

COLORS = ['C{}'.format(i) for i in range(10)]

STAGES = {
    'train': 'Training',
    'valid': 'Validation',
    'test': 'Testing',
    'all_': 'All',
    'cv': 'CV'
}


class OHEncoder:

    def __init__(self, categories=np.array(['A', 'C', 'G', 'T'])):
        self.encoder = Encoder(sparse=False, categories=[categories])
        self.dictionary = categories
        self.encoder.fit(categories.reshape(-1, 1))

    def __call__(self, seq, info=False):
        seq = list(seq)
        if 'N' in seq:
            pos = [i for i, el in enumerate(seq) if el == 'N']
            if len(pos) <= 0.05*len(seq):
                if info:
                    print('{} unknown position(s) in given sequence - changed to random one(s)'.format(len(pos)))
                for p in pos:
                    seq[p] = random.choice(self.dictionary)
            else:
                return None
        if info:
            return True
        else:
            s = np.array(seq).reshape(-1, 1)
            return self.encoder.transform(s).T

    def decode(self, array):
        return ''.join([el[0] for el in self.encoder.inverse_transform(array.T)])


def make_chrstr(chrlist):

    cl = chrlist.copy()
    cl.sort()
    cl.append(0)
    chrstr = ''
    first = cl[0]
    och = first
    for ch in cl:
        if ch == och:
            och += 1
        elif first != och-1:
            if len(chrstr) != 0:
                chrstr += ', '
            chrstr += '%d-%d' % (first, och-1)
            first = ch
            och = ch+1
        else:
            if len(chrstr) != 0:
                chrstr += ', '
            chrstr += '%d' % first
            first = ch
            och = ch+1

    return chrstr


def read_chrstr(chrstr):

    chrstr = chrstr.strip('[]')
    c = chrstr.split(',')
    chrlist = []
    for el in c:
        el = el.split('-')
        if len(el) == 1:
            chrlist.append(int(el[0]))
        else:
            chrlist += [i for i in range(int(el[0]), int(el[1])+1)]
    chrlist.sort()

    return chrlist


def calculate_metrics(confusion_matrix, losses):
    from statistics import mean
    from itertools import product
    num_classes = confusion_matrix.shape[0]
    sens, spec = [], []
    for cl in range(num_classes):
        tp = confusion_matrix[cl][cl]
        fn = sum([confusion_matrix[row][cl] for row in range(num_classes) if row != cl])
        tn = sum([confusion_matrix[row][col] for row, col in product(range(num_classes), repeat=2)
                  if row != cl and col != cl])
        fp = sum([confusion_matrix[cl][col] for col in range(num_classes) if col != cl])
        sens += [float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0]
        spec += [float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0]
    loss = [mean(el) if el else None for el in losses]
    return loss, sens, spec


def calculate_auc(true, scores):
    from sklearn.metrics import roc_auc_score
    num_classes = len(scores[0])
    auc = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for neuron in range(num_classes):
        y_true = [1 if el == neuron else 0 for el in true]
        y_score = [el[neuron] for el in scores]
        if len(set(y_true)) <= 1:
            auc[neuron][neuron] = np.nan
        else:
            auc[neuron][neuron] = roc_auc_score(y_true, y_score)
        for neg in [i for i in range(num_classes) if i != neuron]:
            y_help = [1 if el == neuron else 0 if el == neg else -1 for el in true]
            y_score = [el[neuron] for use, el in zip(y_help, scores) if use != -1]
            y_true = [el for el in y_help if el != -1]
            if len(set(y_true)) <= 1:
                auc[neuron][neg] = np.nan
            else:
                auc[neuron][neg] = roc_auc_score(y_true, y_score)
    return auc


def correct_old_auc_results(file):
    f = open(file, 'r')
    w = open(file.replace('.tsv', '_corrected.tsv'), 'w')
    header = f.readline()
    w.write(header)
    print(header)
    print(header.strip().split('\t'))
    cols = [i for i, el in enumerate(header.strip().split('\t')) if 'AUC' in el]
    print(cols)
    for line in f:
        l = line.strip().split('\t')
        towrite = []
        for i, el in enumerate(l):
            if i in cols:
                el = el.split(', ')
                new = [0 for _ in range(len(cols))]
                new[cols.index(i)] = el[0]
                for v, vv in zip(el[1:], [e for e, ee in enumerate(new) if ee == 0]):
                    new[vv] = v
                towrite.append(', '.join(new))
            else:
                towrite.append(el)
        w.write('\t'.join(towrite) + '\n')
    f.close()
    w.close()


def write_params(glob, file):
    with open(file, 'w') as f:
        for name, value in PARAMS.items():
            v = glob[value]
            if isinstance(v, list):
                if 'num' in value:
                    f.write('{}: {}\n'.format(name, make_chrstr(v)))
                else:
                    f.write('{}: {}\n'.format(name, ''.join(['\n\t{}'.format(el) for el in list(map(str, v))])))
            elif isinstance(v, dict):
                towrite = '{}:'.format(name)
                for key, val in v.items():
                    if isinstance(val, list):
                        val = ', '.join(map(str, val))
                    towrite += '\n\t{}: {}'.format(key, val)
                f.write('{}\n'.format(towrite))
            else:
                f.write('{}: {}\n'.format(name, v))


def basic_params(parser, param=False):
    parser.add_argument('-p', '--path', action='store', metavar='DIR', type=str, default=None,
                        help='Working directory.')
    parser.add_argument('--namespace', action='store', metavar='NAME', type=str, default=None,
                        help='Namespace of the analysis, default: established based on input file')
    parser.add_argument('-o', '--output', action='store', metavar='DIR', type=str, default=None,
                        help='Output directory, default: [PATH]/results/[NAMESPACE]')
    parser.add_argument('--seed', action='store', metavar='NUMBER', type=int, default='0',
                        help='Set random seed, default: 0')
    if param:
        parser.add_argument('--param', action='store', metavar='NAME', type=str, default=None,
                            help='File with parameters of the network, from which results should be plotted, ' +
                                 'if PATH is given, file is supposed to be in PATH directory: [PATH]/[NAME], ' +
                                 'default: [PATH]/[NAMESPACE]_params.txt')
    return parser


def parse_arguments(args, file, namesp=None, model_path=False):
    if args.path is not None:
        path = args.path
    elif file is not None:
        if isinstance(file, list):
            path = os.path.dirname(file[0])
            for f in file[1:]:
                p = os.path.dirname(f)
                path = ''.join([el for el, le in zip(p, path) if el == le])
        else:
            path = os.path.dirname(file)
            if not os.path.isfile(file):
                path = os.path.join(os.getcwd().rstrip('bin'), 'data', 'custom40', path)  # default example dataset of the project
    if path.endswith('data'):
        path = path[:-4]
    if args.namespace is not None:
        try:
            if args.run is not None:
                namespace = args.namespace + args.run
            else:
                namespace = args.namespace
        except AttributeError:
            namespace = args.namespace
    elif namesp is not None:
        namespace = namesp
    elif file is not None:
        f = file if not isinstance(file, list) else file[0]
        _, namespace = os.path.split(f)
        namespace = namespace.split('_')[0]
    else:
        _, namespace = os.path.split(path)
    print('Namespace: {}'.format(namespace))
    if args.output is not None:
        output = args.output
    else:
        if 'results' in path:
            output = path
        else:
            output = os.path.join(path, 'results', namespace)
    if model_path and 'results' not in path:
        path = os.path.join(path, 'results', namespace)
    print('Path: {}'.format(path))
    print('Output directory: {}'.format(output))
    return path, output, namespace, args.seed


def results_header(stage, logger, columns, classes=()):
    if 'AUC-neuron' in columns.keys():
        name, formatting = columns['AUC-neuron']
        del columns['AUC-neuron']
        for i, classname in enumerate(classes):
            columns['AUC - {}'.format(classname)] = [name.replace('INT', str(i)), formatting]
    towrite = '\t'.join(columns.keys())
    if 'train' in stage:
        logger.info('Epoch\tStage\t{}'.format(towrite))
    elif 'test' in stage:
        logger.info('Dataset\tSubset\t{}'.format(towrite))
    return logger, columns


def read_results_columns(results_table, columns):
    file = results_table.handlers[0].baseFilename
    with open(file, 'r') as f:
        header = f.readline().strip().split('\t')
        i = 0
        if header == ['']:
            return []
        for h in header:
            if h not in columns:
                if 'auc' in h.lower():
                    name, formatting = columns['AUC-neuron']
                    columns[h] = [name.replace('INT', str(i)), formatting]
                    i += 1
                elif not h:
                    continue
                elif h not in ['Epoch', 'Stage', 'Dataset', 'Subset']:
                    print(h)
                    raise ValueError
    if i > 0:
        del columns['AUC-neuron']
    return columns


def write_results(logger, columns, stages, variables, *beginning):
    for stage in stages:
        result_string = ''
        for begin in beginning:
            if isinstance(begin, list):
                result_string += '\t{}'.format('; '.join(begin))
            else:
                result_string += '\t{}'.format(begin)
        if 'train' in stage or 'valid' in stage:
            result_string += '\t{}'.format(stage)
        for col, formatting in columns.values():
            if col[-1].isdigit():
                try:
                    variable = variables['{}_{}'.format(stage, col[:-1])][int(col[-1])]
                except TypeError:
                    variable = None
            else:
                variable = variables['{}_{}'.format(stage, col)]
            if variable is None:
                result_string += '\t----'
            elif formatting == 'float-list':
                result_string += '\t' + ', '.join(['{:.2f}'.format(el) if isinstance(el, float) else '{}'.format(el)
                                                   for el in variable])
            elif formatting == 'float':
                result_string += '\t{:.2f}'.format(variables['{}_{}'.format(stage, col)])
        logger.info(result_string.strip())


def write_test_results(logger, columns, variables, data_dir, subset):
    result_string = '{}\t{}'.format(data_dir, subset)
    for col, formatting in columns.values():
        if col[-1].isdigit():
            variable = variables[col[:-1]][int(col[-1])]
        else:
            variable = variables['test_{}'.format(col)]
        if formatting == 'float-list':
            result_string += '\t' + ', '.join(['{:.2f}'.format(el) for el in variable])
        elif formatting == 'float':
            result_string += '\t{:.2f}'.format('test_{}'.format(variable))
    logger.info(result_string)


def check_cuda(logger):
    import torch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        if logger is not None:
            logger.info('--- CUDA available ---')
        else:
            print('--- CUDA available ---')
    else:
        if logger is not None:
            logger.info('--- CUDA not available ---')
        else:
            print('--- CUDA not available ---')
    return use_cuda, device


def build_loggers(stage, output='.', namespace='test', verbose_mode=True, logfile=True, resultfile=True):
    import logging
    formatter = logging.Formatter('%(message)s')
    loggers = []
    if logfile or verbose_mode:
        logger = logging.getLogger('verbose')
        logger.setLevel(logging.INFO)
        loggers.append(logger)
    if verbose_mode:
        cmd_handler = logging.StreamHandler()
        cmd_handler.setFormatter(formatter)
        logger.addHandler(cmd_handler)
    if logfile:
        log_handler = logging.FileHandler(os.path.join(output, '{}_{}.log'.format(namespace, stage)))
        log_handler.setFormatter(formatter)
        logger.addHandler(log_handler)
    if resultfile:
        results_table = logging.getLogger('{}_results'.format(stage))
        results_file = os.path.join(output, '{}_{}_results.tsv'.format(namespace, stage))
        if os.path.isfile(results_file):
            old_results = True
        else:
            old_results = False
        results_handler = logging.FileHandler(results_file)
        results_handler.setFormatter(formatter)
        results_table.addHandler(results_handler)
        results_table.setLevel(logging.INFO)
        loggers.append(results_table)
        return loggers, old_results
    return loggers


def get_classes_names(param):
    with open(param, 'r') as f:
        for line in f:
            if line.startswith('Possible classes'):
                neurons = [el for el in line.split(':')[-1].strip().split('; ') if el]
                l = f.readline()
                while l.startswith('\t'):
                    neurons.append(l.strip())
                    l = f.readline()
                break
    return neurons


def divide_chr(train, val, test):
    import random
    train_chr, val_chr, test_chr = [], [], []
    if not train.startswith('-'):
        train_chr = read_chrstr(train)
    if not val.startswith('-'):
        val_chr = read_chrstr(val)
    if not test.startswith('-'):
        test_chr = read_chrstr(test)
    used_chr = train_chr + val_chr + test_chr
    if not train_chr:
        train_chr = random.sample([el for el in range(1, 24) if el not in used_chr], int(train.lstrip('-')))
        train_chr.sort()
        used_chr += train_chr
    if not val_chr:
        val_chr = random.sample([el for el in range(1, 24) if el not in used_chr], int(val.lstrip('-')))
        val_chr.sort()
        used_chr += val_chr
    if not test_chr:
        test_chr = random.sample([el for el in range(1, 24) if el not in used_chr], int(test.lstrip('-')))
        test_chr.sort()
    return train_chr, val_chr, test_chr


def params_from_file(param_file, data_dir=()):
    seq_len = 2000
    ch = {}
    with open(param_file, 'r') as f:
        for line in f:
            if line.startswith('Possible classes'):
                if len(line.split(';')) > 1:
                    classes = line.split(':')[1].strip().split('; ')
                else:
                    line = f.readline()
                    classes = []
                    while line.startswith('\t'):
                        classes.append(line.strip('\t\n'))
                        line = f.readline()
            if line.startswith('Network type'):
                network = NET_TYPES[line.split(':')[-1].strip().lower()]
            elif line.startswith('Name of the analysis'):
                name = line.split(':')[-1].strip()
            elif line.startswith('Data directory') and not data_dir:
                data_dir = [el for el in line.split(':')[-1].strip().split('; ') if el]
                if not data_dir:
                    l = f.readline()
                    while l.startswith('\t'):
                        data_dir.append(l.strip())
                        l = f.readline()
            elif line.strip().startswith('Input sequence length'):
                seq_len = int(line.split(':')[-1].strip())
            elif line.startswith('Training chr'):
                ch['train'] = read_chrstr(line.split(':')[-1].strip())
            elif line.startswith('Validation chr'):
                ch['valid'] = read_chrstr(line.split(':')[-1].strip())
            elif line.startswith('Test chr'):
                ch['test'] = read_chrstr(line.split(':')[-1].strip())
            elif line.startswith('Number of epochs'):
                num_epochs = int(line.split(':')[-1].strip())
    return network, data_dir, seq_len, ch, classes, name, num_epochs


def read_classes(file):
    _, _, _, _, neurons, _, _ = params_from_file(file)
    return neurons


def validate(model, loader, num_classes, num_batches, use_cuda, output_values=None, logger=None, test=False):
    with torch.no_grad():
        model.eval()
        confusion_matrix = np.zeros((num_classes, num_classes))
        loss_neurons = [[] for _ in range(num_classes)]
        true, scores = [], []
        for i, (seqs, labels) in enumerate(loader):
            if use_cuda:
                seqs = seqs.cuda()
                labels = labels.cuda()
            seqs = seqs.float()
            labels = labels.long()

            outputs = model(seqs)

            for o, l in zip(outputs, labels):
                loss_neurons[l].append(-math.log((math.exp(o[l])) / (sum([math.exp(el) for el in o]))))

            _, indices = torch.max(outputs, axis=1)
            for ind, label, outp in zip(indices, labels.cpu(), outputs):
                confusion_matrix[ind][label] += 1
                if output_values is not None:
                    output_values[label] = [el + [outp[j].cpu().item()] for j, el in
                                            enumerate(output_values[label])]

            true += labels.tolist()
            scores += outputs.tolist()

            if logger is not None and i % 10 == 0:
                logger.info('Batch {}/{}'.format(i, num_batches))

    # Calculate metrics
    losses, sens, spec = calculate_metrics(confusion_matrix, loss_neurons)
    auc = calculate_auc(true, scores)
    if output_values is not None and test:
        return losses, sens, spec, auc, output_values, true, confusion_matrix
    elif output_values is not None and not test:
        return losses, sens, spec, auc, output_values
    elif output_values is None and test:
        return losses, sens, spec, auc, true, confusion_matrix
    elif output_values is None and not test:
        return losses, sens, spec, auc


def print_results_log(logger, stage_desc, classes, stage_sens, stage_spec, stage_auc, class_stage, header=True):
    from statistics import mean
    if header:
        logger.info("{:>35s}{:.5s}, {:.5s}, {:.5s}".format('', 'SENSITIVITY', 'SPECIFICITY', 'AUC'))
    logger.info("--{:>18s} :{:>5} seqs{:>22}".
                format(stage_desc, sum([len(el) for el in class_stage.values()]), "--"))
    for i, (cl, sens, spec, auc) in enumerate(zip(classes, stage_sens, stage_spec, stage_auc)):
        logger.info(
            '{:>20} :{:>5} seqs - {:1.3f}, {:1.3f}, {:1.3f}'.format(cl, len(class_stage[cl]), sens, spec, auc[i]))
    means = []
    for stage_result in [stage_sens, stage_spec, [el[i] for i, el in enumerate(stage_auc) if not np.isnan(el[i])]]:
        if stage_result:
            means.append(mean(stage_result))
        else:
            means.append(np.nan)
    logger.info(
        "--{:>18s} : {:1.3f}, {:1.3f}, {:1.3f}{:>12}\n".
            format('MEANS', *means, "--"))

'''def print_results(logger, columns, variables, epoch):
    logger.info("Epoch {} finished in {:.2f} min\nTrain loss: {:1.3f}\n{:>35s}{:.5s}, {:.5s}"
                .format(epoch + 1, (time() - t0) / 60, train_loss_reduced, '', 'SENSITIVITY', 'SPECIFICITY'))
    logger.info("--{:>18s} :{:>5} seqs{:>15}".format('TRAINING', train_len, "--"))
    for cl, seqs, sens, spec in zip(dataset.classes, data_labels[0], train_sens, train_spec):
        logger.info('{:>20} :{:>5} seqs - {:1.3f}, {:1.3f}'.format(cl, seqs, sens, spec))
    logger.info("--{:>18s} :{:>5} seqs{:>15}".format('VALIDATION', val_len, "--"))
    for cl, seqs, sens, spec in zip(dataset.classes, data_labels[1], val_sens, val_spec):
        logger.info('{:>20} :{:>5} seqs - {:1.3f}, {:1.3f}'.format(cl, seqs, sens, spec))
    logger.info(
        "--{:>18s} : {:1.3f}, {:1.3f}{:>12}".format('TRAINING MEANS', *list(map(mean, [train_sens, train_spec])), "--"))
    logger.info(
        "--{:>18s} : {:1.3f}, {:1.3f}{:>12}\n\n".format('VALIDATION MEANS', *list(map(mean, [val_sens, val_spec])),
                                                        "--"))
    formatter = logging.Formatter('%(message)s')
logger = logging.getLogger('verbose')
results_table = logging.getLogger('results')
cmd_handler = logging.StreamHandler()
log_handler = logging.FileHandler(os.path.join(output, '{}.log'.format(namespace)))
results_handler = logging.FileHandler(os.path.join(output, '{}_results.tsv'.format(namespace)))
for logg, handlers in zip([logger, results_table], [[cmd_handler, log_handler], [results_handler]]):
    for handler in handlers:
        handler.setFormatter(formatter)
        logg.addHandler(handler)
    logg.setLevel(logging.INFO)                                         
                                                        
                                                        
                                                        '''
