import argparse
from bin.common import *

parser = argparse.ArgumentParser(description='Train network based on given data')

parser.add_argument('--model', action='store', metavar='NAME', type=str, default='data/alt1/alt1_last.model',
                    help='File with the model weights to load before training ')

parser.add_argument('--namespace', action='store', metavar='NAME', type=str, default='TEST-RUN',
                    help='The namespace for this run of training')

parser.add_argument('-p', '--path', action='store', metavar='DIR', type=str, default="data/test_fasta",
                        help='Working directory.')
parser.add_argument('-x', '--prefix', action='store', type=str, default="test",
                        help='file_prefix.')

parser.add_argument('-o', '--output', action='store', metavar='DIR', type=str, default="data/test_output",
                        help='Output directory, default: test_output')
parser.add_argument('--seed', action='store', metavar='NUMBER', type=int, default='0',
                        help='Set random seed, default: 0')

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
parser.add_argument('--dropout-conv', action='store', metavar='FLOAT', type=float, default=None,
                    help='Dropout of convolutional layers, default value is 0.2')
parser.add_argument('--dropout-fc', action='store', metavar='FLOAT', type=float, default=None,
                    help='Dropout of fully-connected layers, default value is 0.5')
parser.add_argument('--weight-decay', action='store', metavar='FLOAT', type=float, default=0.0001,
                    help='Weight decay, default value is 0.0001')
parser.add_argument('--momentum', action='store', metavar='FLOAT', type=float, default=0.1,
                    help='Momentum, default value is 0.1')

args = parser.parse_args()



