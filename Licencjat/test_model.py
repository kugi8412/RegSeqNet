import torch
from torch import nn
import math, os, random, csv
import pandas as pd
from Bio import SeqIO
import numpy as np
from sklearn.preprocessing import OneHotEncoder as Encoder
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from statistics import mean
from itertools import product
import numpy as np
import random


class CustomNetwork(torch.nn.Module):

    def __init__(self, seq_len=2000, num_channels=[300, 200, 200], kernel_widths=[19, 11, 7], pooling_widths=[3, 4, 4],
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
        if torch.cuda.is_available():
            self.conv_layers = self.conv_layers.cuda()

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
        if torch.cuda.is_available():
            self.fc_layers = self.fc_layers.cuda()

    def forward(self, x):        
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_input) 
        x = self.fc_layers(x)        
        return torch.sigmoid(x)

def parse_chromosome(seq_id):
        """
        Extracts chromosome number from the sequence ID.
        """
        chromosome = seq_id[1:]

        if chromosome.startswith("chr"):
            chromosome = chromosome[3:]
        
        return "23" if chromosome in ["X", "Y"] else chromosome


def read_sequences(file, filetype="fasta"):
    """
    Funkcja wczytujaca sekwencje z pliku
    
    :param file: plik z sekwencjami
    :param filetype: typ pliku, domyslnie fasta
    :return: sekwencje wczytane z pliku
    """
    
    sequences = []
    
    for seq_record in SeqIO.parse(file, filetype):
        # tylko chromosomy ze zbioru testowego
        if seq_record.id.lower() in ["chr18", "chr19", "chr20", "chr21", "chr22", "chrx", "chry", "chr23"]:
            if 'N' not in seq_record.seq:
                sequences.append(seq_record)
    
    return sequences

class OHEncoder:

    def __init__(self, categories=np.array(['A', 'C', 'G', 'T'])):
        self.encoder = Encoder(sparse_output=False, categories=[categories])
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


def create_dataset():
    """
    Funkcja sluzaca do utworzenia zbioru sekwencji, ktore beda uzyte
    do testowania modeli oryginalnych i zmienionych. Z kazdego zbioru 
    danych (promotor aktywny/nieaktywny, niepromotor aktywny/nieaktywny) 
    wybiera wszystkie sekwencje nalezace do tego zbioru danych.
    Sekwencje nalezace do tego zbioru, zapisuje do pliku.
    """

    np.random.seed(1234567890)
    random.seed(1234567890)
    
    datasets = ['promoter_active', 'nonpromoter_active', 'promoter_inactive', 'nonpromoter_inactive']
    
    dictionary = {}
    lengths = []
    # CHOOSE DATASET
    for dataset in datasets:
        ## TO DO
        # sequences = read_sequences("./data/custom40/{}_alt.fa".format(dataset))
        sequences = read_sequences("./data/custom40/{}_high.fa".format(dataset))
        dictionary[dataset] = sequences
        lengths.append(len(sequences))

    output = []

    for key in dictionary:
        output.extend(dictionary[key])

    shuffled = sorted(output, key=lambda k: random.random())
    
    path = "./test_set/"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "testset.txt", "w") as file:
        for seq in shuffled:
            SeqIO.write(seq, file, 'fasta')
        
    return shuffled


class SeqsDataset(Dataset):
    """
    Klasa sluzaca do utworzenia zbioru testowego, do ktorego naleza sekwencje
    DNA nieuzyte do treningu sieci
    """

    def __init__(self, seq_len=2000):

        np.random.seed(123)

        self.classes = ['promoter active', 'nonpromoter active', 'promoter inactive', 'nonpromoter inactive']
        self.num_classes = len(self.classes)
        self.seq_len = seq_len
        self.encoder = OHEncoder()
        self.data = create_dataset()
        for i in range(len(self.data)):
            _seq, _label = self.__getitem__(i, info=False)
            if _seq is not None and _label is not None:
                seq, label = _seq, _label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index, info=False):
        
        seq = self.data[int(index)].seq.upper()
        id = self.data[int(index)].description.split(" ")
        label = self.classes.index(id[len(id)-5].lower() + " " + id[len(id)-4].lower())
        
        encoded_seq = self.encoder(seq, info=info)
        try:
            X = torch.tensor(encoded_seq)
            X = X.reshape(1, *X.size())
            y = torch.tensor(label)
            return X, y
        except RuntimeError:
            return None, None


def test_network(modelfile, network, name_model):
    """
    Funkcja służąca do testowania modelu na różnych zbiorach danych,
    a następnie rysowania wykresów AUC, Czułości, Swoistości i funkcji kosztu.
    """
    
    batch_size = 64
    classes = ['promoter active', 'nonpromoter active', 'promoter inactive', 'nonpromoter inactive']
    num_classes = len(classes)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = CustomNetwork()
    model.load_state_dict(torch.load(modelfile, map_location=torch.device(device), weights_only=True))
    dataset = SeqsDataset()
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    prediction = []
    true, scores = [], []
    pred_vals = []

    with torch.no_grad():
        model.eval()
        confusion_matrix = np.zeros((num_classes, num_classes))
        loss_neurons = [[] for _ in range(num_classes)]
        path = "./test_set/prediction/diff_prediction/"
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path+"{}_{}_pred_diff.csv".format(network, name_model), "w") as pred_diff_file:
            writer = csv.writer(pred_diff_file)
            writer.writerow(["Tensor", "Seq", "Class", "Prediction", "Pred values {}".format(classes[0]),
                             "Pred values {}".format(classes[1]), "Pred values {}".format(classes[2]),
                             "Pred values {}".format(classes[3])])
            for i, (seqs, labels) in enumerate(testloader):
                if use_cuda:
                    seqs = seqs.cuda()
                    labels = labels.cuda()
                seqs = seqs.float()
                labels = labels.long()
                outputs = model(seqs)
                pred_vals.append(outputs)
                for o, l in zip(outputs, labels):
                    loss_neurons[l].append(-math.log((math.exp(o[l])) / (sum([math.exp(el) for el in o]))))
                _, predicted = torch.max(outputs, 1)
                for ind, label, outp in zip(predicted, labels.cpu(), outputs):
                    confusion_matrix[ind][label] += 1
                prediction.append(predicted)
                # Wariancja predykcji
                for j in range(len(labels.tolist())):
                    if labels.tolist()[j] != predicted.tolist()[j]:
                        out_vals = outputs.tolist()[j]
                        writer.writerow(["tensor_{}".format(i), "seq_{}".format(i*batch_size+j),
                                         classes[labels.tolist()[j]], classes[predicted.tolist()[j]],
                                         out_vals[0], out_vals[1], out_vals[2], out_vals[3]])
                true += labels.tolist()
                scores += outputs.tolist()

   # zapisanie metryk
    path = "./test_set/prediction/{}/metrics/".format(name_model)
    if not os.path.exists(path):
        os.makedirs(path)

    def calculate_metrics(confusion_matrix, losses, true, prediction):
        num_classes = confusion_matrix.shape[0]
        sens, spec, mccs = [], [], []
        
        for cl in range(num_classes):
            tp = confusion_matrix[cl][cl]
            fn = sum([confusion_matrix[row][cl] for row in range(num_classes) if row != cl])
            tn = sum([confusion_matrix[row][col] for row, col in product(range(num_classes), repeat=2) if row != cl and col != cl])
            fp = sum([confusion_matrix[cl][col] for col in range(num_classes) if col != cl])
            
            sens.append(float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0)
            spec.append(float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0)
            mcs = float(tp)*float(tn) - float(fp)*float(fn)
            mcs /=(((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))**0.5)
            mccs.append(mcs if (tp + fn) > 0 else 0.0)
        

        loss = [mean(el) if el else None for el in losses]
        return loss, sens, spec, mccs

    def calculate_auc(true, scores):
        num_classes = len(scores[0])
        auc = [0 for _ in range(num_classes)]
        for neuron in range(num_classes):
            y_true = [1 if el == neuron else 0 for el in true]
            y_score = [el[neuron] for el in scores]
            try:
                auc[neuron] = roc_auc_score(y_true, y_score)
            except:
                pass
        return auc

    # Zapis metryk
    with open(path+"{}_{}_metrics.csv".format(network, name_model), "w") as file_metrics:
        losses, sens, spec, mcc_score = calculate_metrics(confusion_matrix, loss_neurons, true, prediction)
        auc = calculate_auc(true, scores)
        writer = csv.writer(file_metrics)
        writer.writerow(["AUC", "Sensitivity", "Specificity", "Losses", "MCC"])
        writer.writerow([auc, sens, spec, losses, mcc_score])


    # save class
    path = "./test_set/prediction/"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + "{}_{}_pred.txt".format(network, name_model), "w") as class_file:
        for pred in prediction:
            pred_list = pred.tolist()
            class_list = [classes[i] for i in pred_list]
            class_file.write(str(class_list))
            class_file.write("\n")

    # zapisanie wartosci z predykcji
    path = "./test_set/prediction/outputs/"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+"{}_{}_outputs.txt".format(network, name_model), "w") as file_outputs:
        for out in pred_vals:
            file_outputs.write(str(out.tolist()))
            file_outputs.write("\n")


def main():
    # Change directory
    directory = "test_output\cohorts\c11"

    dirs = ["\procent020_1", "\procent020_2", "\procent020_3", "\procent020_4", "\procent020_5",
            "\procent040_1", "\procent040_2", "\procent040_3", "\procent040_4", "\procent040_5"
    ]

    networks = ["origin", "best"]

    for dir in dirs:
        actual_dir = directory + dir
        for network in networks:
            model = "high" + "_" + dir[8:]
            model_file = ".\data\{}\{}_{}.model".format(actual_dir, network, model)
            test_network(model_file, network, model)

        print("Finish test for model")
    
                

if __name__ == "__main__":
    main()
