import torch
import os
from torch import nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import math, os, random
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder as Encoder
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from statistics import mean
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

    with open(path+"testset.txt", "w") as file:
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
            X = torch.tensor(encoded_seq, dtype=torch.float32)
            X = X.reshape(1, *X.size())
            y = torch.tensor(label, dtype=torch.long)
            return X, y
        except RuntimeError:
            return None, None


# Funkcja testująca model
def test_network(modelfile, dataset):
    """
    Testuje model i zwraca średnie wartości AUC oraz MCC.
    """
    batch_size = 64
    num_classes = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Wczytanie modelu
    model = CustomNetwork().to(device)
    model.load_state_dict(torch.load(modelfile, weights_only=True))
    model = model.to(torch.float32)
    model.eval()

    testloader = DataLoader(dataset, batch_size=batch_size)
    true_labels, scores = [], []
    confusion_matrix = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for seqs, labels in testloader:
            seqs, labels = seqs.to(device), labels.to(device)
            outputs = model(seqs)

            _, predicted = torch.max(outputs, 1)
            for pred, label in zip(predicted, labels):
                confusion_matrix[pred][label] += 1
            
            true_labels.extend(labels.cpu().tolist())
            scores.extend(outputs.cpu().tolist())

    def calculate_auc(true, scores):
        auc_values = []
        for i in range(num_classes):
            y_true = [1 if label == i else 0 for label in true]
            y_score = [score[i] for score in scores]
            try:
                auc_values.append(roc_auc_score(y_true, y_score))
            except:
                auc_values.append(0)  
        return np.mean(auc_values)

    def calculate_mcc(conf_matrix):
        mcc_values = []
        for i in range(num_classes):
            tp = conf_matrix[i, i]
            fn = sum(conf_matrix[:, i]) - tp
            fp = sum(conf_matrix[i, :]) - tp
            tn = conf_matrix.sum() - (tp + fn + fp)
            
            numerator = (tp * tn) - (fp * fn)
            denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
            mcc_values.append(numerator / denominator if denominator != 0 else 0)
        return np.mean(mcc_values)

    avg_auc = calculate_auc(true_labels, scores)
    avg_mcc = calculate_mcc(confusion_matrix)

    return avg_auc, avg_mcc

# FOLDERY
dataset = SeqsDataset()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

folder_path = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/"
directories = [folder_path + "data/test_output/cohorts/c2/", 
               folder_path + "data/test_output/cohorts/c3/", 
               folder_path + "data/test_output/cohorts/c4/", 
               folder_path + "data/test_output/cohorts/c5/", 
               folder_path + "data/test_output/cohorts/c6/",
               folder_path + "data/test_output/cohorts/c7/"]

dirs = [
    "procent020_1", "procent020_2", "procent020_3", "procent020_4", "procent020_5",
    "procent040_1", "procent040_2", "procent040_3", "procent040_4", "procent040_5",
    "procent060_1", "procent060_2", "procent060_3", "procent060_4", "procent060_5",
    "procent080_1", "procent080_2", "procent080_3", "procent080_4", "procent080_5"
]

# FILTRY DO ZEROWANIA
groups = {
    "grupa1": [70, 50, 225, 200, 299, 188, 108, 17, 159, 214, 236],
    "grupa2": [236],
    "grupa3": [17],
    "grupa4": [279, 61, 43, 116, 268, 1, 83, 34, 198, 207, 269],
    "grupa5": [166, 224, 122, 257, 47, 111, 294, 297, 269, 85, 131]
}

for directory in directories:
    for sub_dir in dirs:
        model_path = f"{directory}{sub_dir}/best_high_{sub_dir[7:]}.model"
        output_folder = os.path.join(directory, sub_dir, "filtered_metrics")
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(model_path):
            print(f"Model nie znaleziony: {model_path}")
            continue

        # Sprawdzenie oryginalnych wyników
        original_auc, original_mcc = test_network(model_path, dataset)
        with open(os.path.join(output_folder, "original_metrics.txt"), "w") as f:
            f.write(f"AUC: {original_auc:.4f}\nMCC: {original_mcc:.4f}\n")

        # Zmiana filtrów
        for group_name, filter_indices in groups.items():
            modified_model = CustomNetwork().to(device)
            modified_model.load_state_dict(torch.load(model_path, weights_only=True))

            with torch.no_grad():
                for name, param in modified_model.named_parameters():
                    if "conv_layers" in name and "weight" in name:
                        param[filter_indices] = 0  
                        break

            # Zapis pliku
            modified_model_path = os.path.join(output_folder, f"model_{group_name}.pth")
            torch.save(modified_model.state_dict(), modified_model_path)

            # Testowanie zmodyfikowanego modelu
            mod_auc, mod_mcc = test_network(modified_model_path, dataset)

            # Zapis wyników dla grupy
            with open(os.path.join(output_folder, f"metrics_{group_name}.txt"), "w") as f:
                f.write(f"AUC: {mod_auc:.4f}\nMCC: {mod_mcc:.4f}\n")

            print(f"Przetestowano {sub_dir} dla {group_name} -> AUC: {mod_auc:.4f}, MCC: {mod_mcc:.4f}")

