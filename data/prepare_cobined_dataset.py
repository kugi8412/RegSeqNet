import random
import os
from Bio import SeqIO
import pandas as pd
import math

path = '/home/marni/magisterka/data/patient_specific'
common_path = '/home/marni/magisterka/data'
ref_model = 'custom40'
seq_files = ['patient_specific_thresh2_pos_subset.fasta', 'patient_specific_thresh2_neg_subset.fasta',
             'patient_specific_thresh2_ref_subset.fasta']

# Establish reference seqs - valid and test from custom40
# ref_seqs = open('../../results/{0}/{0}_valid.txt'.format(ref_model), 'r').read().strip().split('\n') + \
#                open('../../results/{0}/{0}_test.txt'.format(ref_model), 'r').read().strip().split('\n')
ref_seqs = open('/home/marni/magisterka/results/{0}/{0}_train.txt'.format(ref_model), 'r').read().strip().split('\n')
print('number of ref_seqs = {}'.format(len(ref_seqs)))
assert len(ref_seqs) == len(set(ref_seqs))
classes = {'promoter active': 0,
           'promoter inactive': 0,
           'nonpromoter active': 0,
           'nonpromoter inactive': 0}

towrite = ''
spec_seqs = set()
for file in seq_files:
    with open(os.path.join(path, file), 'r') as f:
        for line in f:
            if line.startswith('>'):
                line_class = [el for el in classes.keys() if el in line]
                header = line.strip().split(' ')
                spec_seqs.add('{}:{}'.format(header[1].lstrip('chr'), header[2]))
                assert len(line_class) == 1
                classes[line_class[0]] += 1
            towrite += line
print(classes)

loc = pd.read_csv(os.path.join(common_path, 'TSSs_1000_flanked_transcripts_ids_genes_ids.bed'), header=None, sep='\t')
loc.set_index(3, inplace=True)
loc.columns = ['chr', 'start', 'end', 'idk', 'strand', 'gene']
loc.drop('idk', axis=1, inplace=True)
loc = loc[~loc.duplicated(['chr', 'start'], keep='first')]

for num_seqs in [20000, 40000]:
    seqs_per_class = int(num_seqs / 4)
    print('Group {}, seqs per class {}'.format(num_seqs, seqs_per_class))
    new_file = 'patient_specific_thresh2_{}.fasta'.format(num_seqs)
    w = open(os.path.join(path, new_file), 'w')
    w.write(towrite)
    for group, already_seqs in classes.items():
        input_file = "/home/marni/magisterka/data/dataset3/{}_10000.fa".format(group.replace(' ', '_'))
        todraw = []
        for i, record in enumerate(SeqIO.parse(input_file, "fasta")):
            header = record.description.strip().split(' ')
            idd = '{}:{}'.format(header[0].lstrip('chr'), header[1])
            if idd not in spec_seqs:
                todraw.append(i)
        seqs_to_draw = seqs_per_class - already_seqs
        print('{} seqs draw from {} ({} already is)'.format(seqs_to_draw, seqs_per_class, already_seqs))
        subset = random.sample(todraw, k=seqs_to_draw)
        for i, record in enumerate(SeqIO.parse(input_file, "fasta")):
            if i in subset:
                header = record.description.strip().split(' ')
                idd = '{}:{}'.format(header[0].lstrip('chr'), header[1])
                '''toheader = None
                loc_clip = loc[(loc['chr'] == header[0]) & (loc['start'] > int(header[1]) - 1500) &
                               (loc['end'] < int(header[1]) + 1500)]
                for n, row in loc_clip.iterrows():
                    if (((row['end'] - row['start']) % 2 == 0 and row['start'] == 2 * int(header[1]) - row['end']) or
                       ((row['end'] - row['start']) % 2 == 1 and row['start'] == 2 * int(header[1]) - row['end'] - 1)):
                        toheader = '{} REF'.format(n)
                if toheader is None:
                    print(header)
                    print(loc[(loc['chr'] == header[0]) & (loc['start'] > int(header[1]) - 10000) &
                               (loc['end'] < int(header[1]) + 10000)])
                    print(loc[(loc['chr'] == header[0]) & (loc['start'] > int(header[1]) - 50000) &
                              (loc['end'] < int(header[1]) + 50000)])
                    raise ValueError
                w.write('>{} {}\n{}\n'.format(record.description, toheader, record.seq))'''
                w.write('>{} {} REF\n{}\n'.format(record.description, idd, record.seq))
    print(new_file)
    w.close()
