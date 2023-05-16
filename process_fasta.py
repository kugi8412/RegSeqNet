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
from Bio import SeqIO
from nn_util import *       

"""

Część własna, daje wynik
Podajemy komendę w postaci >> python seq_in.py model lista_nazw_seq.txt
""" 

parser = argparse.ArgumentParser(description='Program pomocniczy podający wektor 4 neuronów dla podanego modelu i zbioru sekwencji')

parser.add_argument('-m','--model', nargs=1, help="Gotowy model, który chcemy wyorzystać do wyznaczenia wartości neurona wyników.", default=None)
parser.add_argument('-i','--ins', nargs=1,  help="Niepustu plik tekstowy fasta zawierający sekwencje nukleotydów w przyjętej formie.",default=None)
parser.add_argument('-o','--out', nargs=1, help="Nazwa pliku tekstowego, do którego program zapisze wyniki podane przez modekl i prawdziwy typ sekwencji.",default=None)

args = parser.parse_args()

if args.model is None:
	print("!!!	Podaj model\n")
elif os.path.isfile(args.model[0]):
	print(">> poprawnie podany model\n")
	modelfile = args.model[0]
	if '/' in modelfile:
		model_param=modelfile.split("/")[-1]
		name=model_param
else:
	print("!!!	Podana ścieżka nie jest poprawna lub plik nie istnieje\n")




if args.ins is None:
	print("!	Podaj plik z listą sekwencji\n")
elif os.path.isfile(args.ins[0]) or os.path.abspath(args.ins[0]):
    data_in = SeqIO.parse(args.ins[0],"fasta")
else:
	print("!!!	Podana ścieżka nie jest poprawna lub plik nie istnieje\n")



if args.out is None:
	data_out=f'{args.ins[0].split(".")[0]}_out.txt'
	out = open(data_out,'w')
elif os.path.isfile(args.out[0]) or os.path.isfile(os.path.abspath(args.out[0])) :
	print("podany plik out istnieje\n")
#	data_out = f"{args.out[0].split('.')[0]}_{name.split('_')[0]}.txt"
	data_out = args.out[0]
	out = open(data_out,'r+')
elif not re.search("//",args.out[0]):
	data_out = args.out[0]
	out = open(data_out,'w')
else:
	print("!!!	Podana ścieżka nie jest poprawna lub plik nie istnieje\n")



#model_param=f"{modelfile.split('/',-1)[0]}/{model_param.split('_')[0]}_params.txt"

#par=open(os.path.normpath(model_param))
#for line in par:
#	if line.startswith('Network type'):
#		network = NET_TYPES[line.split(':')[-1].strip().lower()]
#par.close()
network=NET_TYPES["custom"]
model = network(2000)
model.load_state_dict(torch.load(modelfile, map_location=torch.device('cpu')))
model.eval()


OHE=OHEncoder()

for seqRec in data_in:
    seq=seqRec.seq
    if len(seq)!=2000:
        print("Problem with sequence",seqRec.id,len(seq))
    else:    
        encoded_seq=OHE(seq)
        if encoded_seq is None:
            print("Too many Ns in ",seqRec.id,)
        else:
            X = torch.tensor(encoded_seq[0])
            X = X.reshape(1,1, *X.size())
            X=X.float()
            y=model(X)
            y=y[0].detach().numpy().tolist()
            y=str(y)
            if encoded_seq[1]:
                out.write(f"{seqRec.id}\t{y}\tOK\n")
            else:
                out.write(f"{seqRec.id}\t{y}\tzawiera_N\n")


out.close()
		
