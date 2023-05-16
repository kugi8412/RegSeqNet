from sklearn.preprocessing import OneHotEncoder as Encoder
import argparse
import numpy as np
import torch
import re
import os
import random
import torch.nn as nn
import torch.nn.functional as F
import math

"""

Sekcja najważniejszych klas zebranych w jednym miejscu

W większości skopiowane wprost z githuba

"""


class OHEncoder:

	def __init__(self, categories=np.array(['A', 'C', 'G', 'T'])):
		self.encoder = Encoder(sparse=False, categories=[categories])
		self.dictionary = categories
		self.encoder.fit(categories.reshape(-1, 1))
	def __call__(self,seq):
		seq = list(seq)
		info = 1
		if 'N' in seq:
			pos = [i for i, el in enumerate(seq) if el == 'N']
			if len(pos) <= 0.05*len(seq):
				info=0
				print('{} unknown position(s) in given sequence - changed to random one(s)'.format(len(pos)))
				for p in pos:
					seq[p] = random.choice(self.dictionary)
			else:
				return None
		s = np.array(seq).reshape(-1, 1)
		return self.encoder.transform(s).T, info







class BassetNetwork(torch.nn.Module):

    def __init__(self, seq_len):
        super(BassetNetwork, self).__init__()
        dropout = 0.3
        pooling_widths = [3, 4, 4]
        num_channels = [300, 200, 200]
        kernel_widths = [19, 11, 7]
        paddings = [int((w-1)/2) for w in kernel_widths]
        num_units = [1000, 4]
        self.params = {'input sequence length': seq_len,'dropout': dropout}

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, num_channels[0], kernel_size=(4, kernel_widths[0]), padding=(0, paddings[0])),
            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[0]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[0])

        self.layer2 = nn.Sequential(
            nn.Conv2d(num_channels[0], num_channels[1], kernel_size=(1, kernel_widths[1]), padding=(0, paddings[1])),
            nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[1]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[1])

        self.layer3 = nn.Sequential(
            nn.Conv2d(num_channels[1], num_channels[2], kernel_size=(1, kernel_widths[2]), padding=(0, paddings[2])),
            nn.BatchNorm2d(num_channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[2]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[2])

        self.fc_input = 1 * seq_len * num_channels[-1]
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.fc_input, out_features=num_units[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout))

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=num_units[0], out_features=num_units[1]),
            nn.ReLU(),
            nn.Dropout(p=dropout))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, self.fc_input)  # reshape
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


class CustomNetwork(torch.nn.Module):

    def __init__(self, seq_len, num_channels=[300, 200, 200], kernel_widths=[19, 11, 7], pooling_widths=[3, 4, 4],
                 num_units=[2000, 4], dropout=0.5):
        super(CustomNetwork, self).__init__()
        paddings = [int((w-1)/2) for w in kernel_widths]
        self.seq_len = seq_len
        self.dropout = dropout
        self.params = {
            'input sequence length': seq_len,
            'convolutional layers': len(num_channels),
            'fully connected': len(num_units),
            'number of channels': num_channels,
            'kernels widths': kernel_widths,
            'pooling widths': pooling_widths,
            'units in fc': num_units,
            'dropout': dropout

        }

        conv_modules = []
        num_channels = [1] + num_channels
        for num, (input_channels, output_channels, kernel, padding, pooling) in \
                enumerate(zip(num_channels[:-1], num_channels[1:], kernel_widths, paddings, pooling_widths)):
            k = 4 if num == 0 else 1
            conv_modules += [
                nn.Conv2d(input_channels, output_channels, kernel_size=(k, kernel), padding=(0, padding)),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, pooling), ceil_mode=True)
            ]
            seq_len = math.ceil(seq_len / pooling)
        self.conv_layers = nn.Sequential(*conv_modules)

        fc_modules = []
        self.fc_input = 1 * seq_len * num_channels[-1]
        num_units = [self.fc_input] + num_units
        for input_units, output_units in zip(num_units[:-1], num_units[1:]):
            fc_modules += [
                nn.Linear(in_features=input_units, out_features=output_units),
                nn.ReLU(),
                nn.Dropout(p=self.dropout)
            ]
        self.fc_layers = nn.Sequential(*fc_modules)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_input)  # reshape
        x = self.fc_layers(x)
        return torch.sigmoid(x)


class TestNetwork(torch.nn.Module):

    def __init__(self, seq_len):
        super(TestNetwork, self).__init__()
        dropout = 0.3
        pooling_widths = [3, 4, 4]
        num_channels = [30, 20, 20]
        kernel_widths = [5, 3, 3]
        paddings = [int((w-1)/2) for w in kernel_widths]
        num_units = [100, 4]
        self.params = {
            'Input sequence length': seq_len,
            'Dropout': dropout
        }

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, num_channels[0], kernel_size=(4, kernel_widths[0]), padding=(0, paddings[0])),
            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[0]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[0])

        self.layer2 = nn.Sequential(
            nn.Conv2d(num_channels[0], num_channels[1], kernel_size=(1, kernel_widths[1]), padding=(0, paddings[1])),
            nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[1]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[1])

        self.layer3 = nn.Sequential(
            nn.Conv2d(num_channels[1], num_channels[2], kernel_size=(1, kernel_widths[2]), padding=(0, paddings[2])),
            nn.BatchNorm2d(num_channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[2]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[2])

        self.fc_input = 1 * seq_len * num_channels[-1]
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.fc_input, out_features=num_units[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout))

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=num_units[0], out_features=num_units[1]),
            nn.ReLU(),
            nn.Dropout(p=dropout))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, self.fc_input)  # reshape
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


class DropoutNetwork(torch.nn.Module):

	def __init__(self, seq_len, num_channels=[300, 200, 200], kernel_widths=[19, 11, 7], pooling_widths=[3, 4, 4],
			num_units=[2000, 4], dropout_conv=0.2, dropout_fc=0.5):
		super(DropoutNetwork, self).__init__()
		paddings = [int((w-1)/2) for w in kernel_widths]
		self.seq_len = seq_len
		self.dropout_conv = dropout_conv
		self.dropout_fc = dropout_fc
		self.params = {
			'input sequence length': seq_len,
			'convolutional layers': len(num_channels),
			'fully connected': len(num_units),
			'number of channels': num_channels,
			'kernels widths': kernel_widths,
			'pooling widths': pooling_widths,
			'units in fc': num_units,
			'dropout conv': dropout_conv,
			'dropout fc': dropout_fc

		}

		conv_modules = []
		num_channels = [1] + num_channels
		for num, (input_channels, output_channels, kernel, padding, pooling) in \
				enumerate(zip(num_channels[:-1], num_channels[1:], kernel_widths, paddings, pooling_widths)):
			k = 4 if num == 0 else 1
			conv_modules += [
				nn.Conv2d(input_channels, output_channels, kernel_size=(k, kernel), padding=(0, padding)),
				nn.BatchNorm2d(output_channels),
				nn.ReLU(),
				nn.Dropout2d(p=self.dropout_conv),
				nn.MaxPool2d(kernel_size=(1, pooling), ceil_mode=True)
			]
			seq_len = math.ceil(seq_len / pooling)
		self.conv_layers = nn.Sequential(*conv_modules)

		fc_modules = []
		self.fc_input = 1 * seq_len * num_channels[-1]
		num_units = [self.fc_input] + num_units
		for input_units, output_units in zip(num_units[:-1], num_units[1:]):
			fc_modules += [
				nn.Linear(in_features=input_units, out_features=output_units),
				nn.ReLU(),
				nn.Dropout(p=self.dropout_fc)
			]
		self.fc_layers = nn.Sequential(*fc_modules)

	def forward(self, x):
		x = self.conv_layers(x)
		x = x.view(-1, self.fc_input)  # reshape
		x = self.fc_layers(x)
		return torch.sigmoid(x)
        
        
        
NET_TYPES = {'basset': BassetNetwork,'custom': CustomNetwork,'test': TestNetwork,'dropout': DropoutNetwork, 'random-forest': 'RandomForest'}       


