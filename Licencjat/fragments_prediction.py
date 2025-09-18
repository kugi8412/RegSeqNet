import torch
from torch import nn
import math, os, random, csv, glob
import pandas as pd
from Bio import SeqIO
import numpy as np
from sklearn.preprocessing import OneHotEncoder as Encoder
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from statistics import mean
from itertools import product
import numpy as np
from scipy.stats import gaussian_kde
import random

from analyse_filters import generate_filter_alt
from analyse_filters import change_index


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

def filters_from_file(file_name):
    filters = []
    with open(file_name, 'r') as file:
        for line in file:
            if not line.startswith(">filter_"):
                try:
                    numbers = [float(x) for x in line.strip().split()]
                    filters.extend(numbers)
                except ValueError:
                    print("Problem with convert '{}' to float.".format(line.strip()))
    return filters

def change_weights(modelfile, dataset, network, ic, filters, m, prob=1.90):
    """
    Funkcja sluzaca do wyzerowania wag w pozycjach filtrow, ktore odpowiadaja
    7-merow o wysokim/niskim ic. Zapisuje zmienione modele do pliku.
    
    :param modelfile: wyuczona siec, w ktorej nalezy wyzerowac filtry
    :param dataset: rozpatrywana klasa
    :param network: rozpatrywana siec
    :param ic: parametr okreslajacy czy wyzerowane sa fragmenty o niskim 
    :param filter: filtr, ktorego wagi nalezy zmienic
    """
    
    path = "./ic_filters/prediction/{}/{}/models_{}/".format(dataset, ic.lower(), m)
    if not os.path.exists(path):
        os.makedirs(path)
    
    # wartosci ic dla kazdego 7-meru badanych filtrow
    #df = pd.read_csv("./ic_filters/{}/{}.csv".format(dataset, network))
    # fragmenty o niskich lub wysokich wartosciach ic
    if "low" in ic.lower():
        # df = pd.read_csv("./statistics/{}{}_{}_filters_below_tresh.csv".format(dataset, network, m))
        df = pd.read_csv("./statistics/alt1_last_filters_below_tresh.csv")
        vals = df.iloc[:, 2].values
        fff = df.iloc[:, 1].values
        nums = [fff[i].split('_', 1)[1] for i in range(len(fff))]
        nums = list(map(int, nums))
        boolean = False
    elif "high" in ic.lower():
        # df = pd.read_csv("./statistics/{}{}_{}_filters_above_tresh.csv".format(dataset, network, m))
        df = pd.read_csv("./statistics/alt1_last_filters_above_tresh.csv")
        vals = df.iloc[:, 2].values
        fff = df.iloc[:, 1].values
        nums = [fff[i].split('_', 1)[1] for i in range(len(fff))]
        boolean = True
    else:
        # df = pd.read_csv("./statistics/{}{}_{}_stats.csv".format(dataset, network, m))
        df = pd.read_csv("./statistics/alt1_last_stats.csv")
        return None
    # wartosci ic wybranego filtra i albo te o wysokim, albo o niskim ic
    # df_filer = df[(df["Filter"] == "filter_{}".format(filter))]
    # indeksy, ktore nalezy wyzerowac (np. 0_8 oznacza, ze wyzerowane beda 
    # indeksy od 0 do 7 wlacznie)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = CustomNetwork()
    
    # wczytanie wytrenowanego modelu
    model.load_state_dict(torch.load(modelfile, map_location=torch.device(device), weights_only=True))
    for name, param in model.named_parameters():
        # zerowa warstwa konwolucyjna
        if name == "conv_layers.0.weight":
            weights = param.detach()
            break
    
    print(nums)
    probability = prob # set probability
    # wyzerowanie w filtrach wag, ktorych pozycje w macierzy
    # odpowiadaja fragmentowi macierzy ppm o niskim/wysokim ic
    ''''
    weights_changed = weights
    for filter in filters:
        if str(filter) not in nums:
            if random.random() < probability:
                for i in range(len(weights_changed[filter][0])):
                    for idx in indices:
                        ranges = [int(x) for x in idx.split("_")]
                        for j in range(ranges[0], ranges[1]):
                            new_value = kde.resample(1)[0]
                            if torch.cuda.is_available():
                                new_value = torch.from_numpy(new_value).cuda(0)
                            weights_changed[filter][0, i, j] = new_value  # zmiana wagi filtrów
    '''
    weights_changed = weights
    """
    for filter in filters:
        if str(filter) not in nums:
            if random.random() < probability:
                weights_changed[filter] = torch.from_numpy(generate_filter_alt()).cuda(0)
        else:
             weights_changed[filter] = torch.from_numpy(np.zeros((4, 19)).cuda(0))
    """

    for filt_idx in filters:
        if str(filt_idx) in nums and random.random() < probability:
            old_idx = filt_idx
            new_idx = None
            while new_idx is None or str(new_idx) in nums:
                new_idx = random.choice(filters)

            weights_changed[new_idx] = weights_changed[old_idx]
            weights_changed[old_idx] = torch.zeros((4, 19), device='cuda:0')

    # podstawienie pod siec zmienionych filtrow
    for name, param in model.named_parameters():
        if name == "conv_layers.0.weight":
            param.data.copy_(weights_changed)
            
    # zapisanie zmienionej sieci
    torch.save(model.state_dict(), path+network+"_{}.model".format(ic))


def read_sequences(file, filetype = "fasta"):
    """
    Funkcja wczytujaca sekwencje z pliku
    
    :param file: plik z sekwencjami
    :param filetype: typ pliku, domyslnie fasta
    :return: sekwencje wczytane z pliku
    """
    
    sequences = []
    
    for seq_record in SeqIO.parse(file, filetype):
        # tylko chromosomy ze zbioru testowego
        if seq_record.id.lower() in ["chr21", "chr22", "chrx", "chry", "chr23"]:
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

    # np.random.seed(1234567890)
    # random.seed(1234567890)
    
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
    
    path = "./ic_filters/"
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

        # np.random.seed(123)

        self.classes = ['promoter active', 'nonpromoter active', 'promoter inactive', 'nonpromoter inactive']
        self.num_classes = len(self.classes)
        self.seq_len = seq_len
        self.encoder = OHEncoder()
        self.data = create_dataset()
        for i in range(len(self.data)):
            seq, label = self.__getitem__(i, info=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index, info=False):

        seq = self.data[int(index)].seq.upper()
        id = self.data[int(index)].description.split(" ")
        label = self.classes.index(id[len(id)-5].lower() + " " + id[len(id)-4].lower())
        
        encoded_seq = self.encoder(seq, info=info)
        
        X = torch.tensor(encoded_seq)
        X = X.reshape(1, *X.size())
        y = torch.tensor(label)
        return X, y



def test_network(modelfile, d, network, ic, filter, m):
    """
    Funkcja sluzaca do testowania modeli zmienionych i oryginalnego. Tworzy wyniki 
    predykcji i zapisuje je do pliku. Sprawdza wahania miedzy predykcjami. Wykonuje 
    ocene predykcji za pomoca AUC, czulosci, specyficznosci i straty. 
    
    :param modelfile: model, ktory zostanie bedzie sluzyl do predykcji
    :param d: rozpatrywana klasa
    :param network: rozpatrywana siec
    :param ic: parametr okreslajacy czy wyzerowane sa fragmenty o niskim 
    czy wysokim ic 
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
        path = "./ic_filters/prediction/{}/diff_prediction/".format(ic.lower())
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path+"{}_{}_pred_diff.csv".format(network, m), "w") as pred_diff_file:
            writer = csv.writer(pred_diff_file)
            writer.writerow(["Tensor", "Seq", "Class", "Prediction", "Pred values {}".format(classes[0]), "Pred values {}".format(classes[1]), "Pred values {}".format(classes[2]), "Pred values {}".format(classes[3])])
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
                # zapisanie wahan predykcji w obrebie modelu
                for j in range(len(labels.tolist())):
                    if labels.tolist()[j] != predicted.tolist()[j]:
                        out_vals = outputs.tolist()[j]
                        writer.writerow(["tensor_{}".format(i), "seq_{}".format(i*batch_size+j), classes[labels.tolist()[j]], classes[predicted.tolist()[j]], out_vals[0], out_vals[1], out_vals[2], out_vals[3]])
                true += labels.tolist()
                scores += outputs.tolist()

   # zapisanie metryk
    path = "./ic_filters/prediction/{}/metrics/".format(ic.lower())
    if not os.path.exists(path):
        os.makedirs(path)

    def calculate_metrics(confusion_matrix, losses):
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

    with open(path+"{}_{}_metrics.csv".format(network, m), "w") as file_metrics:
        losses, sens, spec = calculate_metrics(confusion_matrix, loss_neurons)
        auc = calculate_auc(true, scores)        
        writer = csv.writer(file_metrics)
        writer.writerow(["AUC", "Sensitivity", "Specificity", "Losses"])
        writer.writerow([auc, sens, spec, losses])


    # zapisanie przypisanych klas
    path = "./ic_filters/prediction/{}/".format(ic.lower())
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+"{}_{}_pred.txt".format(network, m), "w") as class_file:
        for pred in prediction:
            pred_list = pred.tolist()
            class_list = [classes[i] for i in pred_list]
            class_file.write(str(class_list))
            class_file.write("\n")

    # zapisanie wartosci z predykcji
    path = "./ic_filters/prediction/{}/outputs/".format(ic.lower())
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+"{}_{}_outputs.txt".format(network, m), "w") as file_outputs:
        for out in pred_vals:
            file_outputs.write(str(out.tolist()))
            file_outputs.write("\n")

    

def main():

    """
    datasets = [""]
    networks = ["altnew_filtry_high", "altnew_high"]
    netdir = "altnew"
    models = ["best", "last"]
    """

    """
    df = pd.read_csv("./TEST-RUN_last_filters_below_tresh.csv")
    vals = df.iloc[:,2].values
    filters = df.iloc[:,1].values
    nums = [filters[i].split('_', 1)[1] for i in range(len(filters))]
    """

    datasets = [""]
    net_dir = ""
    networks = ["alt1"]
    models = ["last"]


    for m in models:
        print(m)
        for dataset in datasets:
            print(dataset)
            for network in networks:
                print(network)
                filters = [[i for i in range(300)]]
                for filter in filters:
                    print(filter)
                    ics = ["original", "low_to_zero", "high_to_zero"]
                    ics = ["high_to_zero"]
                    for ic in ics:
                        print(ic)
                        # wyzerowanie wag we fragmentach o wybranym ic
                        change_weights(".\data\{}\{}_{}.model".format(net_dir, network, m), dataset, network, ic, filter, m)
                        # wykonanie predykcji 
                        if ic == "original":
                            model_file = ".\data\{}\{}_{}.model".format(net_dir, network, m)
                        else:
                            model_file = "./ic_filters/prediction/{}/models_{}/{}_{}.model".format(ic.lower(), m, network, ic.lower())
                        test_network(model_file, dataset, network, ic, filter, m)

        print("Finish")
                
    
if __name__ == "__main__":
    # Set seed is prob_prob_trial_prop_prop*trial*chohort[5]**3 e.g. 202012020*1*1**8 for c1
    # Set seed prob_trial_prop * trail (cohort + 10)**3
    # for 4th cohort momentum is 0.02
    # c5 is the same as c4 with difference in momentum
    # w c8 seed zamieniony z prob_prob z 20 z 80
    # random_seed = 1001100 * 1 * (11+10)**3
    # 8080* 1 * (11+10)**3
    # random_seed = 8080* 1 * (11+10)**3
    random_seed = 42 * 100
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    main()
