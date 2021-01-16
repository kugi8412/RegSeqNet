from bin.datasets import SeqsDataset
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
from bin.common import *
from bin.networks import *
import math
import os
from statistics import mean
from time import time
from datetime import datetime
import numpy as np
import shutil
from collections import OrderedDict
import random

from bin.common import NET_TYPES

OPTIMIZERS = {
    'RMSprop': optim.RMSprop,
    'Adam': optim.Adam
}

LOSS_FUNCTIONS = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'MSELoss': nn.MSELoss
}


RESULTS_COLS = OrderedDict({
    'Loss': ['losses', 'float-list'],
    'Sensitivity': ['sens', 'float-list'],
    'Specificity': ['spec', 'float-list'],
    'AUC-neuron': ['aucINT', 'float-list']
})


def adjust_learning_rate(lr, epoch, optimizer):
    if epoch > 500:
        lr = lr / 100000
    elif epoch > 400:
        lr = lr / 10000
    elif epoch > 300:
        lr = lr / 1000
    elif epoch > 200:
        lr = lr / 100
    elif epoch > 100:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


parser = argparse.ArgumentParser(description='Train network based on given data')
parser.add_argument('data', action='store', metavar='DATASET', type=str, nargs='+',
                    help='Folder with the data for training and validation, if PATH is given, data is supposed to be ' +
                         'in PATH directory: [PATH]/data/[DATA]')
parser.add_argument('-n', '--network', action='store', metavar='NAME', type=str, default='basset',
                    help='type of the network to train, default: Basset Network')
parser = basic_params(parser)
parser.add_argument('--run', action='store', metavar='NUMBER', type=str, default=None,
                    help='number of the analysis, by default NAMESPACE is set to [NETWORK][RUN]')
parser.add_argument('--train', action='store', metavar='NUM', type=int, default=None,
                    help='Number of sequences for training')
parser.add_argument('--valid', action='store', metavar='NUM', type=int, default=None,
                    help='Number of sequences for validation')
parser.add_argument('--test', action='store', metavar='NUM', type=int, default=None,
                    help='Number of sequences for testing')
parser.add_argument('--train_chr', action='store', metavar='CHR', type=str, default='1-16',
                    help='chromosome(s) for training, if negative it means the number of chromosomes ' +
                         'which should be randomly chosen. Default: 1-16')
parser.add_argument('--valid_chr', action='store', metavar='CHR', type=str, default='17-20',
                    help='chromosome(s) for validation, if negative it means the number of chromosomes ' +
                         'which should be randomly chosen. Default: 17-20')
parser.add_argument('--test_chr', action='store', metavar='CHR', type=str, default='21-23',
                    help='chromosome(s) for testing, if negative it means the number of chromosomes ' +
                         'which should be randomly chosen. Default: 21-23')
parser.add_argument('--optimizer', action='store', metavar='NAME', type=str, default='RMSprop',
                    help='optimization algorithm to use for training the network, default = RMSprop')
parser.add_argument('--loss_fn', action='store', metavar='NAME', type=str, default='CrossEntropyLoss',
                    help='loss function for training the network, default = CrossEntropyLoss')
parser.add_argument('--batch_size', action='store', metavar='INT', type=int, default=64,
                    help='size of the batch, default: 64')
parser.add_argument('--num_workers', action='store', metavar='INT', type=int, default=4,
                    help='how many subprocesses to use for data loading, default: 4')
parser.add_argument('--num_epochs', action='store', metavar='INT', type=int, default=300,
                    help='maximum number of epochs to run, default: 300')
parser.add_argument('--acc_threshold', action='store', metavar='FLOAT', type=float, default=0.9,
                    help='threshold of the validation accuracy - if gained training process stops, default: 0.9')
parser.add_argument('--learning_rate', action='store', metavar='FLOAT', type=float, default=0.01,
                    help='initial learning rate, default: 0.01')
parser.add_argument('--no_adjust_lr', action='store_true',
                    help='no reduction of learning rate during training, default: False')
parser.add_argument('--seq_len', action='store', metavar='INT', type=int, default=2000,
                    help='Length of the input sequences to the network, default: 2000')
parser.add_argument('--model', action='store', metavar='NAME', type=str, default=None,
                    help='File with the model weights to load before training, if PATH is given, '
                         'model is supposed to be in PATH directory, '
                         'if NAMESPACE is given model is supposed to be in [PATH]/results/[NAMESPACE]/ directory')
parser.add_argument('--constant_class', action='store', metavar='CLASS', type=str, default=None,
                    help='If all sequences from the given dataset should belong to given class')
parser.add_argument('--dropout', action='store', metavar='FLOAT', type=float, default=None,
                    help='Dropout for training (available only for Custom network), default value is 0.5')
parser.add_argument('--check_the_subset', action='store', metavar='FILE', type=str, nargs='+', default=None,
                    help='File with list of IDs of sequences that should be used for additional validation '
                         'during training')
parser.add_argument('--name_pos', action='store', metavar='INT', nargs='+', default=None,
                    help='Position(s) of sequence name in the fasta header, by default created as CHR:POSITION')
args = parser.parse_args()

batch_size, num_workers, num_epochs, acc_threshold, seq_len = args.batch_size, args.num_workers, args.num_epochs, \
                                                              args.acc_threshold, args.seq_len
if args.run is None:
    namesp = args.network + '0'
else:
    namesp = args.network + args.run
path, output, namespace, seed = parse_arguments(args, args.data, namesp=namesp)
# create folder for the output files
if os.path.isdir(output):
    shutil.rmtree(output)
try:
    os.mkdir(output)
except FileNotFoundError:
    os.mkdir(os.path.join(path, 'results'))
    os.mkdir(output)
# establish data directories
if args.path is not None:
    data_dir = [os.path.join(path, 'data', d) for d in args.data]
else:
    data_dir = args.data
    if os.path.isdir(data_dir[0]):
        path = data_dir[0]
# set the random seed
torch.manual_seed(seed)
np.random.seed(seed)
# set other params
network_name = args.network
optimizer_name = args.optimizer
lossfn_name = args.loss_fn
network = NET_TYPES[network_name.lower()]
optim_method = OPTIMIZERS[optimizer_name]
lossfn = LOSS_FUNCTIONS[lossfn_name]
lr = args.learning_rate
weight_decay = 0.0001
if args.no_adjust_lr:
    adjust_lr = False
else:
    adjust_lr = True
if args.model is None:
    modelfile = None
else:
    if os.path.isfile(args.model):
        modelfile = args.model
    else:
        modelfile = os.path.join(output, args.model)
    if args.namespace is None:
        namespace += '-retrain'

# Define files for logs and for results
[logger, results_table], old_results = build_loggers('train', output=output, namespace=namespace)

logger.info('\nAnalysis {} begins {}\nInput data: {}\nOutput directory: {}\n'.format(
    namespace, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), '; '.join(data_dir), output))

t0 = time()
# CUDA for PyTorch
use_cuda, device = check_cuda(logger)

dataset = SeqsDataset(data_dir, seq_len=seq_len, constant_class=args.constant_class, name_pos=args.name_pos)
num_classes = dataset.num_classes
classes = dataset.classes

# write header of results table
if not old_results:
    results_table, columns = results_header('train', results_table, RESULTS_COLS, classes)
else:
    columns = read_results_columns(results_table, RESULTS_COLS)

# Creating data indices for training, validation and test splits:
if not (args.train is not None or args.valid is not None or args.test is not None):
    train_num, valid_num, test_num = divide_chr(args.train_chr, args.valid_chr, args.test_chr)
    if set(train_num) & set(valid_num):
        logger.warning('WARNING - Chromosomes for training and validation overlap!')
    elif set(train_num) & set(test_num):
        logger.warning('WARNING - Chromosomes for training and testing overlap!')
    elif set(valid_num) & set(test_num):
        logger.warning('WARNING - Chromosomes for validation and testing overlap!')
    train_indices, valid_indices, test_indices = dataset.get_chrs([train_num, valid_num, test_num])
else:
    train_num, valid_num, test_num = None, None, None
    if args.train is not None:
        train_num = args.train
    if args.valid is not None:
        valid_num = args.valid
    if args.test is not None:
        test_num = args.test
    if train_num is None:
        if valid_num is None:
            train_num = (dataset.num_seqs - test_num) // 2
        elif test_num is None:
            train_num = (dataset.num_seqs - valid_num) // 2
        else:
            train_num = dataset.num_seqs - valid_num - test_num
    if valid_num is None:
        if test_num is None:
            valid_num = (dataset.num_seqs - train_num) // 2
        else:
            valid_num = dataset.num_seqs - train_num - test_num
    if test_num is None:
        test_num = dataset.num_seqs - valid_num - train_num

    if train_num + valid_num + test_num > dataset.num_seqs:
        print('Number of train, valid and test sequences need to sum up to {}'.format(dataset.num_seqs))
        raise ValueError
    train_indices = random.sample(range(dataset.num_seqs), train_num)
    valid_indices = random.sample([i for i in range(dataset.num_seqs) if i not in train_indices], valid_num)
    test_indices = random.sample([i for i in range(dataset.num_seqs) if i not in train_indices and i not in
                                  valid_indices], test_num)
    indices = [train_indices, valid_indices, test_indices]
class_stage = [dataset.get_classes(el) for el in [train_indices, valid_indices, test_indices]]
train_len, valid_len = len(train_indices), len(valid_indices)
num_seqs = ' + '.join([str(len(el)) for el in [train_indices, valid_indices, test_indices]])
if not (args.train is not None or args.valid is not None or args.test is not None):
    chr_string = ['({})'.format(el) for el in map(make_chrstr, [train_num, valid_num, test_num])]
else:
    chr_string = ['', '', '']
for i, (n, ch, ind) in enumerate(zip(['train', 'valid', 'test'], chr_string,
                                     [train_indices, valid_indices, test_indices])):
    logger.info('\n{} set contains {} seqs {}:'.format(n, len(ind), ch))
    for classname, el in class_stage[i].items():
        logger.info('{} - {}'.format(classname, len(el)))
    # Writing IDs for each split into file
    with open(os.path.join(output, '{}_{}.txt'.format(namespace, n)), 'w') as f:
        f.write('\n'.join([dataset.IDs[j] for j in ind]))

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

logger.info('\nTraining and validation datasets built in {:.2f} s'.format(time() - t0))

if args.check_the_subset is not None:
    subset_ids = []
    for names_file in args.check_the_subset:
        if not os.path.isfile(names_file):
            names_dir, _ = os.path.split(data_dir[0])
            names_file = os.path.join(names_dir, names_file)
        with open(names_file, 'r') as f:
            subset_ids += f.read().strip().split('\n')
            logger.info('Check the subset: sequences names read from {}'.format(names_file))
    subset_ids = set(subset_ids)
    logger.info('Read {} sequences to check'.format(len(subset_ids)))
    subset_indices = dataset.get_indices(subset_ids)
    subset_train_indices = [el for el in subset_indices if el in train_indices]
    subset_valid_indices = [el for el in subset_indices if el in valid_indices]
    subset_train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                      sampler=SubsetRandomSampler(subset_train_indices))
    subset_valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                      sampler=SubsetRandomSampler(subset_valid_indices))
    subset_class_stage = [dataset.get_classes(el) for el in [subset_train_indices, subset_valid_indices]]
    [subset_results_table], subset_old_results = build_loggers('subset-train', output=output, namespace=namespace,
                                                               verbose_mode=False, logfile=False, resultfile=True)
    if not subset_old_results:
        subset_results_table, subset_columns = results_header('subset-train', subset_results_table, RESULTS_COLS, classes)
    else:
        subset_columns = read_results_columns(subset_results_table, RESULTS_COLS)

num_batches = math.ceil(train_len / batch_size)

model = network(dataset.seq_len)
if modelfile is not None:
    # Load weights from the file
    t0 = time()
    model.load_state_dict(torch.load(modelfile, map_location=torch.device(device)))
    logger.info('\nModel from {} loaded in {:.2f} s'.format(modelfile, time() - t0))
if network_name.lower() == 'custom' and args.dropout is not None:
    network.dropout = args.dropout
    logger.info('\nDropout changed to {}'.format(args.dropout))
network_params = model.params
optimizer = optim_method(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = lossfn()
best_acc = 0.0
# write parameters values into file
write_params(globals(), os.path.join(output, '{}_params.txt'.format(namespace)))
logger.info('\n--- TRAINING ---\nEpoch 0 is a data validation without training step')
t = time()
for epoch in range(num_epochs+1):
    t0 = time()
    confusion_matrix = np.zeros((num_classes, num_classes))
    train_loss_neurons = [[] for _ in range(num_classes)]
    train_loss_reduced = 0.0
    true, scores = [], []
    if epoch == num_epochs:
        train_output_values = [[[] for _ in range(num_classes)] for _ in range(num_classes)]
        valid_output_values = [[[] for _ in range(num_classes)] for _ in range(num_classes)]
    for i, (seqs, labels) in enumerate(train_loader):
        if use_cuda:
            seqs = seqs.cuda()
            labels = labels.cuda()
            model.cuda()
        seqs = seqs.float()
        labels = labels.long()

        if epoch != 0:
            model.train()
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            outputs = model(seqs)
            losses = []
            for o, l in zip(outputs, labels):
                loss = -math.log((math.exp(o[l]))/(sum([math.exp(el) for el in o])))
                train_loss_neurons[l].append(loss)
                losses.append(loss)
            train_loss_reduced += mean(losses)

            _, indices = torch.max(outputs, axis=1)
            for ind, label, outp in zip(indices, labels.cpu(), outputs):
                confusion_matrix[ind][label] += 1
                if epoch == num_epochs:
                    train_output_values[label] = [el + [outp[j].cpu().item()] for j, el in enumerate(train_output_values[label])]

            true += labels.tolist()
            scores += outputs.tolist()

        if i % 10 == 0:
            logger.info('Epoch {}, batch {}/{}'.format(epoch, i, num_batches))

    # Call the learning rate adjustment function
    if not args.no_adjust_lr:
        adjust_learning_rate(lr, epoch, optimizer)

    # Calculate metrics
    train_losses, train_sens, train_spec = calculate_metrics(confusion_matrix, train_loss_neurons)
    train_loss_reduced = train_loss_reduced / num_batches
    assert math.floor(mean([el for el in train_losses if el])*10/10) == math.floor(float(train_loss_reduced)*10/10)
    train_auc = calculate_auc(true, scores)

    if epoch == num_epochs:
        valid_losses, valid_sens, valid_spec, valid_auc, valid_output_values = \
            validate(model, valid_loader, num_classes, num_batches, use_cuda, output_values=valid_output_values)
    else:
        valid_losses, valid_sens, valid_spec, valid_auc = \
            validate(model, valid_loader, num_classes, num_batches, use_cuda)

    if args.check_the_subset is not None:
        subset_train_losses, subset_train_sens, subset_train_spec, subset_train_auc = \
            validate(model, subset_train_loader, num_classes, num_batches, use_cuda)
        subset_valid_losses, subset_valid_sens, subset_valid_spec, subset_valid_auc = \
            validate(model, subset_valid_loader, num_classes, num_batches, use_cuda)
        write_results(subset_results_table, subset_columns, ['subset_train', 'subset_valid'], globals(), epoch)

    # Save the model if the test acc is greater than our current best
    if mean(valid_sens) > best_acc and epoch < num_epochs:
        torch.save(model.state_dict(), os.path.join(output, "{}_{}.model".format(namespace, epoch + 1)))
        best_acc = mean(valid_sens)

    # If it is a last epoch write neurons' outputs, labels and model
    if epoch == num_epochs:
        logger.info('Last epoch - writing neurons outputs for each class!')
        np.save(os.path.join(output, '{}_train_outputs'.format(namespace)), np.array(train_output_values))
        np.save(os.path.join(output, '{}_valid_outputs'.format(namespace)), np.array(valid_output_values))
        torch.save(model.state_dict(), os.path.join(output, '{}_last.model'.format(namespace)))

    # Write the results
    write_results(results_table, columns, ['train', 'valid'], globals(), epoch)
    # Print the metrics
    logger.info("Epoch {} finished in {:.2f} min\nTrain loss: {:1.3f}"
                .format(epoch, (time() - t0)/60, train_loss_reduced))

    print_results_log(logger, 'TRAINING', dataset.classes, train_sens, train_spec, train_auc, class_stage[0])
    print_results_log(logger, 'VALIDATION', dataset.classes, valid_sens, valid_spec, valid_auc, class_stage[1], header=False)
    if args.check_the_subset is not None:
        print_results_log(logger, 'TRAINING-SUBSET', dataset.classes, subset_train_sens, subset_train_spec,
                          subset_train_auc, subset_class_stage[0], header=False)
        print_results_log(logger, 'VALIDATION-SUBSET', dataset.classes, subset_valid_sens, subset_valid_spec,
                          subset_valid_auc, subset_class_stage[1], header=False)

    if mean(valid_sens) >= acc_threshold:
        logger.info('Validation accuracy threshold reached!')
        break

logger.info('Training for {} finished in {:.2f} min'.format(namespace, (time() - t)/60))
