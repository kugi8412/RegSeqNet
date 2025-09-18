import os
import torch
import numpy as np
from typing import List, Dict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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
        sequences = read_sequences("./data/custom40/{}_alt.fa".format(dataset))
        # sequences = read_sequences("./data/custom40/{}_high.fa".format(dataset))
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

def load_filters(filepath: str) -> List[np.ndarray]:
    filters, current = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>filter_'):
                if current:
                    filters.append(np.array(current))
                current = []
            elif line:
                current.append(list(map(float, line.split())))
        if current:
            filters.append(np.array(current))
    return filters

# Compute L2 norms

def compute_l2_norms(filters: List[np.ndarray]) -> np.ndarray:
    return np.array([np.linalg.norm(filt, ord='fro') for filt in filters])

# Test network: AUC and MCC
def test_network(
    modelfile: str,
    dataset,
    layer: int,
    idx: int
) -> Dict[int, Dict[str, float]]:
    """
    Testuje model z wyzerowanym filtrem `idx` na warstwie `layer`.
    Zwraca słownik:
       class_i -> {'auc': ..., 'mcc': ...}
    """
    batch_size = 64
    num_classes = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load and prepare the model
    model = CustomNetwork().to(device)
    model.load_state_dict(torch.load(modelfile, map_location=device, weights_only=True))
    model.eval()

    # Zero out the specific filter in the specified layer

    with torch.no_grad():
        # print(model.conv_layers[layer].weight.data[idx])
        conv_layer = model.conv_layers[layer]
        conv_layer.weight.data[idx].zero_().detach()
        # TO CHANGE BATCHNORM
        # print(model.conv_layers[layer].weight.data[idx])

    # Prepare the data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    true_labels, scores = [], []
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Perform predictions
    with torch.no_grad():
        for seqs, labels in loader:
            seqs, labels = seqs.to(device), labels.to(device)
            outputs = model(seqs)
            preds = outputs.argmax(1)

            # Update confusion matrix
            for p, t in zip(preds, labels):
                confusion_matrix[p.item(), t.item()] += 1

            true_labels.extend(labels.cpu().tolist())
            scores.extend(outputs.cpu().tolist())

    # Calculate AUC and MCC for each class
    results = {}
    for i in range(num_classes):
        # AUC for class i
        y_true = [1 if lab == i else 0 for lab in true_labels]
        y_score = [s[i] for s in scores]
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = 0.0

        # MCC for class i
        tp = confusion_matrix[i, i]
        fn = confusion_matrix[:, i].sum() - tp
        fp = confusion_matrix[i, :].sum() - tp
        tn = confusion_matrix.sum() - (tp + fn + fp)
        num = tp * tn - fp * fn
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (num / den) if den != 0 else 0.0

        results[i] = {'auc': auc, 'mcc': mcc}

    return results


# Główna

def main(
    filters_folder: str,
    models_folder: str,
    cohorts: List[str],
    threshold: float,
    dataset
):
    all_results = {}

    for cohort in cohorts:
        # 1) Ścieżka do pliku filtrów best_*_filter.txt
        # best_filter_file = f"best_{cohort.replace('procent', 'high_')}_filter.txt"
        # best_filter_path = os.path.join(filters_folder, best_filter_file)

        # 2) Wczytanie filtrów i wybór indeksów powyżej progu
        # filters = load_filters(folder_dir)
        filters = load_filters(filters_folder)
        idxs = [
            i for i, f in enumerate(filters)
            if np.linalg.norm(f, ord='fro') > threshold
        ]

        # 3) Znalezienie pierwszego pliku .model zaczynającego się od 'best'
        """
        model_cohort_dir = os.path.join(models_folder, cohort)
        best_model_path = None
        for fname in os.listdir(model_cohort_dir):
            if fname.startswith('best') and fname.endswith('.model'):
                best_model_path = os.path.join(model_cohort_dir, fname)
                break
        if best_model_path is None:
            print(f"[ERROR] Brak pliku best*.model w {model_cohort_dir}")
            continue"""

        # 4) Dla każdego filtra testujemy model best
        cohort_results = {}
        for idx in idxs:
            print(f"Testing cohort={cohort}, filter_idx={idx}")
            res_best = test_network(models_folder, dataset, layer=0, idx=idx)
            cohort_results[idx] = res_best

        # 5) Zapis wyników do pliku per kohorta
        out_txt = os.path.join("C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/data", f"impact_best_{cohort}.txt")
        with open(out_txt, 'w') as fw:
            fw.write(f"Threshold (Frobenius): {threshold}\n")
            fw.write(f"Model: {os.path.basename(models_folder)}\n\n")
            for idx, res in cohort_results.items():
                fw.write(f"Filter {idx}:\n")
                for cls, metrics in res.items():
                    fw.write(
                        f"Class {cls}: AUC={metrics['auc']:.4f}, MCC={metrics['mcc']:.4f}\n"
                    )
                fw.write("\n")
        print(f"Saved results to {out_txt}")

        all_results[cohort] = cohort_results

    return all_results


if __name__ == '__main__':
    folder_dir = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/data/test_output/cohorts/c10/filter"
    model_dir = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/data/test_output/cohorts/c10"
    cohorts = [f"procent{prefix}_{i}" for prefix in ["020", "040", "060", "080"] for i in range(1, 6)]
    cohorts = [f"procent{prefix}_{i}" for prefix in ["020"] for i in range(1, 2)]
    folder_dir = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/filters/highsignal_last/high_signal4_last_filter.txt"
    model_dir = "C:/Nauka/Studia/Projekty/Projekt-BW/Zadanie 1/Projekt/LearnModel/.venv/data/high_last.model"
    cohorts = [1]
    threshold = -1.05  # Przykładowy próg
    main(folder_dir, model_dir, cohorts, threshold, SeqsDataset())
