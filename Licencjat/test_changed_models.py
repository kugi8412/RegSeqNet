import torch
from torch import nn
import math
import os
import pandas as pd
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder as Encoder
from torch.utils.data import Dataset
import numpy as np
import random
import csv
import glob

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


def change_weights(modelfile, m):
    """
    Funkcja sluzaca do wyzerowania wag w filtrach pierwszej warstwy,
    ktore maja wartosc sredniej wag ponizej zalozonego progu.
    
    :param modelfile: wyuczona siec, w ktorej nalezy wyzerowac filtry
    """
    
    path = "./changed_models/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    network_name = os.path.basename(modelfile).replace("_{}.model".format(m), "")
        
    # wczytanie filtrow, ktore maja srednia wag ponizej progu
    df = pd.read_csv("./statistics/{}_{}_filters_below_tresh.csv".format(network_name, m))
    below_tresh = df["Filter"].tolist()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = CustomNetwork()
    # wczytanie wytrenowanego modelu
    model.load_state_dict(torch.load(modelfile, map_location=torch.device(device), weights_only=True))
    for name, param in model.named_parameters():
        # zerowa warstwa konwolucyjna
        if name == "conv_layers.0.weight":
            weights = param.detach()

    # wyzerowanie wag filtrow 
    weights_changed = weights
    for i in range(len(weights_changed)):
        if "filter_{}".format(i) in below_tresh:
            weights_changed[i] = torch.zeros(weights_changed[i].size())
            
    # podstawienie pod siec zmienionych filtrow           
    for name, param in model.named_parameters():
        if name == "conv_layers.0.weight":
            param = weights_changed
            
    # zapisanie zmienionej sieci
    torch.save(model.state_dict(), path+network_name+"_changed_{}.model".format(m))


def read_sequences(file, filetype="fasta"):
    """
    Funkcja wczytujaca sekwencje z pliku
    
    :param file: plik z sekwencjami
    :param filetype: typ pliku, domyslnie fasta
    """
    
    sequences = []
    
    for seq_record in SeqIO.parse(file, filetype):
        # tylko chromosomy ze zbioru testowego
        if seq_record.id in ["chr21", "chr22", "chrX", "chrY"]:
            print(seq_record)
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


def create_dataset(m):
    """
    Funkcja sluzaca do utworzenia zbioru sekwencji, ktore beda uzyte
    do testowania modeli oryginalnych i zmienionych. Z kazdego zbioru 
    danych (promotor aktywny/nieaktywny, niepromotor aktywny/nieaktywny) 
    wybiera wszystkie sekwencje nalezace do tego zbioru danych.
    Sekwencje nalezace do tego zbioru, zapisuje do pliku.
    """

    np.random.seed(123)
    random.seed(123)
    
    datasets = ['promoter_active', 'nonpromoter_active', 'promoter_inactive', 'nonpromoter_inactive']
    
    dictionary = {}
    lengths = []

    for dataset in datasets:
        sequences = read_sequences(".\data\custom40\{}_alt.fa".format(dataset))
        dictionary[dataset] = sequences
        lengths.append(len(sequences))

    output = []

    for key in dictionary:
        output.extend(dictionary[key])

    shuffled = sorted(output, key=lambda k: random.random())
    
    path = './prediction_{}/'.format(m)
    if not os.path.exists(path):
        os.makedirs(path)

    with open("./prediction_{}/testset.txt".format(m), "w") as file:
        for seq in shuffled:
            SeqIO.write(seq, file, 'fasta')
        
    return shuffled


class SeqsDataset(Dataset):

    def __init__(self, m, seq_len=2000):

        np.random.seed(123)

        self.classes = ['promoter active', 'nonpromoter active', 'promoter inactive', 'nonpromoter inactive']
        self.num_classes = len(self.classes)
        self.seq_len = seq_len
        self.encoder = OHEncoder()
        self.data = create_dataset(m)
        for i in range(len(self.data)):
            seq, label = self.__getitem__(i, info=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index, info=False):

        seq = self.data[int(index)].seq.upper()
        id = self.data[int(index)].description.split(" ")
        label = self.classes.index(id[len(id)-5]+" "+id[len(id)-4])
        
        encoded_seq = self.encoder(seq, info=info)
        X = torch.tensor(encoded_seq)
        X = X.reshape(1, *X.size())
        y = torch.tensor(label)
        return X, y


def test_network(modelfile, m, changed = False):
    """
    Funkcja sluzaca do testowania modelu oryginalnego i zmienionego.
    Tworzy wyniki predykcji i zapisuje je do pliku. Sprawdza
    wahania miedzy predykcjami. Wykonuje ocene predykcji za pomoca
    AUC itp. Porownuje predykcje dla modelu zmienionego i oryginalnego
    
    :param modelfile: model, ktory zostanie bedzie sluzyl do predykcji
    """
    if changed:
        networkname = os.path.basename(modelfile).replace(".model", "")
    else:
        networkname = os.path.basename(modelfile).replace("_{}.model".format(m), "")
    
    batch_size = 64

    classes = ['promoter active', 'nonpromoter active', 'promoter inactive', 'nonpromoter inactive']

    num_classes = len(classes)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = CustomNetwork()
    model.load_state_dict(torch.load(modelfile, map_location=torch.device(device), weights_only=True))

    dataset = SeqsDataset(m)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    prediction = []
    true, scores = [], []
    pred_vals = []
    with torch.no_grad():
        model.eval()
        confusion_matrix = np.zeros((num_classes, num_classes))
        loss_neurons = [[] for _ in range(num_classes)]
        # zapisanie wahań predykcji w obrębie modelu
        path = "./prediction_{}/diff_prediction/".format(m)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path+"{}_pred_diff.csv".format(networkname), "w") as pred_diff_file:
            writer = csv.writer(pred_diff_file)
            writer.writerow(["Tensor", "Seq", "Class", "Prediction", "Prediction values ({},{},{},{})".format(classes[0], classes[1], classes[2], classes[3])])
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
                for j in range(len(labels.tolist())):
                    if labels.tolist()[j] != predicted.tolist()[j]:
                        writer.writerow(["tensor_{}".format(i), "seq_{}".format(i*batch_size+j), classes[labels.tolist()[j]], classes[predicted.tolist()[j]], outputs.tolist()[j]])
                true += labels.tolist()
                scores += outputs.tolist()
                
   # zapisanie metryk
    path = "./prediction_{}/metrics/".format(m)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+"{}_metrics.csv".format(networkname), "w") as file_metrics:
        losses, sens, spec = calculate_metrics(confusion_matrix, loss_neurons)
        auc = calculate_auc(true, scores)
        writer = csv.writer(file_metrics)
        writer.writerow(["Losses", "Sensitivity", "Specificity", "AUC"])
        writer.writerow([losses, sens, spec, auc])
    
    # zapisanie wartosci z predykcji
    path = "./prediction_{}/outputs/".format(m)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+"{}_outputs.txt".format(networkname), "w") as file_outputs:
        for out in pred_vals:
            file_outputs.write(str(out.tolist()))
            file_outputs.write("\n")

    # zapisanie wyjsciowych predycji 
    path = "./prediction_{}/".format(m)
    if not os.path.exists(path):
        os.makedirs(path)
    save_to_file(prediction, path+"{}.txt".format(networkname))


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
    auc = [0 for _ in range(num_classes)]
    for neuron in range(num_classes):
        y_true = [1 if el == neuron else 0 for el in true]
        y_score = [el[neuron] for el in scores]
        auc[neuron] = roc_auc_score(y_true, y_score)
    return auc


def calculate_auc2(true, scores):
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

def save_to_file(prediction, filename):

    classes = ['promoter active', 'nonpromoter active', 'promoter inactive', 'nonpromoter inactive']
    
    with open(filename, "w") as file:
        for pred in prediction:
            pred_list = pred.tolist()
            class_list = [classes[i] for i in pred_list]
            file.write(str(class_list))
            file.write("\n")


def merge_files_metrics():

    classes = ["nonpromoter_active", "nonpromoter_inactive", "promoter_active", "promoter_inactive"]
    with open("./prediction/all_metrics.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "Network", "Losses", "Sensitivity", "Specificity", "AUC"])
        for c in classes:
            directory = "./prediction/{}/metrics/".format(c)
            list_of_files = sorted(filter(os.path.isfile, glob.glob(directory + '*') ) )
            for file_path in list_of_files:
                name_out = file_path.split("/")[len(file_path.split("/"))-1].replace("_metrics.csv", "")
                df1 = pd.read_csv(file_path)
                for index, row in df1.iterrows():
                    data = [row["Losses"], row["Sensitivity"], row["Specificity"], row["AUC"]]
                data.insert(0, name_out)
                data.insert(0, c)
                writer.writerow(data)


def main():
    '''
    networks = ["altnew_filtry_high", "altnew_high"]
    netdir = "altnew"
    models = ["best", "last"]
    '''

    net_dir = "test_output"
    networks = ["best09"]
    models = ["highsignal"]

    for m in models:
        # wyzerowanie wag filtrow dla kazdej sieci
        for network in networks:
            change_weights(".\data\{}\{}_{}.model".format(net_dir, network, m), m)

        for network in networks:
            print(network)
            test_network(".\data\{}\{}_{}.model".format(net_dir, network, m), m)
            test_network(".\changed_models\{}_changed_{}.model".format(network, m), m, changed=True)


if __name__ == "__main__":
    main()
    print("Finish")
