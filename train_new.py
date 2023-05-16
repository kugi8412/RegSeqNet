from new_dataset import SeqsDataset
import torch

from torch.utils import data

from torch.utils.data.sampler import SubsetRandomSampler

from bin.common import *
from bin.networks import *
import math
import os
from statistics import mean
from time import time
from datetime import datetime
import numpy as np
import shutil

import random
from train_new_tools import *
from new_parser import *

namespace="ATAC-Seq"

batch_size, num_workers, num_epochs, acc_threshold, seq_len = args.batch_size, args.num_workers, args.num_epochs, args.acc_threshold, args.seq_len

seed=args.seed
#path, output, namespace, seed = parse_arguments(args, namesp=namespace)


print(args)

# set the random seed
torch.manual_seed(seed)
np.random.seed(seed)
# set other params
network_name = "custom"
optimizer_name = args.optimizer
lossfn_name = args.loss_fn
network = NET_TYPES[network_name.lower()]
optim_method = OPTIMIZERS[optimizer_name]
lossfn = LOSS_FUNCTIONS[lossfn_name]
lr = args.learning_rate
weight_decay = args.weight_decay
momentum = args.momentum
if args.no_adjust_lr:
    adjust_lr = False
else:
    adjust_lr = True

    
if os.path.isfile(args.model):
    modelfile = args.model
    

# Define files for logs and for results
[logger, results_table], old_results = build_loggers('train', output=args.output, namespace=namespace)

logger.info('\nAnalysis {} begins {}\nInput data: {}\nOutput directory: {}\n'.format(
    namespace, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), args.path, args.output))

t0 = time()
# CUDA for PyTorch
use_cuda, device = check_cuda(logger)

dataset = SeqsDataset(f_prefix=args.path+"/"+args.prefix)
num_classes = dataset.num_classes
classes = dataset.classes

# write header of results table
if not old_results:
    results_table, columns = results_header('train', results_table, RESULTS_COLS, classes)
else:
    columns = read_results_columns(results_table, RESULTS_COLS)

# Creating data indices for training, validation and test splits:
train_ids, valid_ids, test_ids = dataset.train_ids, dataset.valid_ids, dataset.test_ids
indices = [train_ids, valid_ids, test_ids]


class_stage = [dataset.get_classes(el) for el in indices]
train_len, valid_len = len(train_ids), len(valid_ids)

num_seqs = ' + '.join([str(len(el)) for el in [train_ids, valid_ids, test_ids]])
chr_string = ['', '', '']
for i, (n, ch, ind) in enumerate(zip(['train', 'valid', 'test'], chr_string,
                                     [train_ids, valid_ids, test_ids])):
    logger.info('\n{} set contains {} seqs {}:'.format(n, len(ind), ch))
    for classname, el in class_stage[i].items():
        logger.info('{} - {}'.format(classname, len(el)))
    # Writing IDs for each split into file
    with open(os.path.join(args.output, '{}_{}.txt'.format(namespace, n)), 'w') as f:
        f.write('\n'.join([dataset.info[j] for j in ind]))

train_sampler = SubsetRandomSampler(train_ids)
valid_sampler = SubsetRandomSampler(valid_ids)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

logger.info('\nTraining and validation datasets built in {:.2f} s'.format(time() - t0))


num_batches = math.ceil(train_len / batch_size)

model = network(dataset.seq_len)
if modelfile is not None:
    # Load weights from the file
    t0 = time()
    model.load_state_dict(torch.load(modelfile, map_location=torch.device(device)))
    logger.info('\nModel from {} loaded in {:.2f} s'.format(modelfile, time() - t0))
if network_name.lower() != 'basset':
    if args.dropout_fc is not None:
        network.dropout_fc = args.dropout_fc
        logger.info('\nDropout-fc changed to {}'.format(args.dropout_fc))
    if args.dropout_conv is not None:
        network.dropout_conv = args.dropout_conv
        logger.info('\nDropout-conv changed to {}'.format(args.dropout_conv))
network_params = model.params
optimizer = optim_method(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
loss_fn = lossfn()
best_acc = 0.0
# write parameters values into file
#write_params(globals(), os.path.join(args.output, '{}_params.txt'.format(namespace)))
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


    # Save the model if the test acc is greater than our current best
    if mean(valid_sens) > best_acc and epoch < num_epochs:
        torch.save(model.state_dict(), os.path.join(args.output, "{}_{}.model".format(namespace, epoch + 1)))
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

    if mean(valid_sens) >= acc_threshold:
        logger.info('Validation accuracy threshold reached!')
        break

logger.info('Training for {} finished in {:.2f} min'.format(namespace, (time() - t)/60))
