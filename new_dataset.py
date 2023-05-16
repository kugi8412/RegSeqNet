from bin.common import OHEncoder
from torch.utils.data import Dataset
import torch
import os
from bin.exceptions import *
from warnings import warn
from rewrite_fasta import rewrite_fasta
import math
from Bio import SeqIO
CLASSES=['promoter active', 'nonpromoter active', 'promoter inactive', 'nonpromoter inactive']
class_sufixes={CLASSES[0]:"_pa.fa",CLASSES[1]:"_na.fa", CLASSES[2]:"_pi.fa",CLASSES[3]:"_ni.fa"}

TRAIN_CHR=["1","2","3","4","5","6","7","8","9","10","11","12"]
VALID_CHR=["13","14","15","16","17"]
TEST_CHR=["18","19","20","21","22"]


class SeqsDataset(Dataset):
    """
    New SeqsDatasetClass for training from ATAC data
    
    f_prefix is a prefix for 4 fasta_files with names prefix_pa.fa prefix_na.fa prefix_pi.fa and prefix_ni.fa
    valid_chroms and test chroms are validation and test chromosomes - will be set aside

    """
    def __init__(self,f_prefix="ATAC" ,train_chr=TRAIN_CHR,valid_chr=VALID_CHR,test_chr=TEST_CHR):
        
        self.classes = CLASSES 
        self.num_classes = len(self.classes)
        self.seqs_per_class = {el: [] for el in self.classes}
        self.ids=[]
        self.train_ids=[]
        self.test_ids=[]
        self.valid_ids=[]
        self.tensors={}
        self.info={}
        self.seq_len=2000


        OHE=OHEncoder()
        curr_id=0
        EncSeqs = {}
        for cl,sufix in class_sufixes.items():
            for SR in SeqIO.parse(f_prefix+sufix,"fasta"):
                encoded_seq=OHE(SR.seq)
                if not (encoded_seq is None) and len(SR.seq)==2000:
                    X=torch.tensor(encoded_seq)
                    X=X.reshape(1,*X.size())
                    y=torch.tensor(CLASSES.index(cl))
                    chrom=SR.id.split(":")[-2]
                    self.info[curr_id]=SR.id
                    if chrom in train_chr:
                        self.ids.append(curr_id)
                        self.train_ids.append(curr_id)
                        self.tensors[curr_id]=X,y
                        self.seqs_per_class[cl].append(curr_id)
                        curr_id+=1
                    elif chrom in valid_chr:
                        self.ids.append(curr_id)
                        self.valid_ids.append(curr_id)
                        self.tensors[curr_id]=X,y
                        self.seqs_per_class[cl].append(curr_id)
                        curr_id+=1
                    elif chrom in test_chr:
                        self.ids.append(curr_id)
                        self.test_ids.append(curr_id)
                        self.tensors[curr_id]=X,y
                        self.seqs_per_class[cl].append(curr_id)
                        curr_id+=1
                    else:
                        print("wrong chromosome",repr(chrom),SR.id,chrom in TRAIN_CHR)
                else: # not encoded
                    print("problem with seq",SR.id,"in class",cl)
                    

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.tensors[index]
        
    def get_classes(self, indices=None):
        if indices is None:
            return self.seqs_per_class
        else:
            return {key: [el for el in value if el in indices] for key, value in self.seqs_per_class.items()}
