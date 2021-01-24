import numpy as np
import torch
import random
import os
from bin.common import OHEncoder
from Bio import SeqIO


def produce_balanced_baseline(outdir, name, num_seq, n=3):
    from itertools import product
    from math import ceil
    print('Establishing balanced baseline named {}, num_seq = {}, n = {}'.format(name, num_seq, n))
    trials = 4**n
    encoder = OHEncoder()
    d = encoder.dictionary
    ntuples = [''.join(el) for el in product(d, repeat=n)]
    presence = [[[] for _ in range(ceil(2000/n))] for _ in range(num_seq)]
    base = []
    for _ in range(trials):
        b = np.zeros((num_seq, 4, 2000))
        for j in range(num_seq):
            choice = [random.choice([el for el in range(len(ntuples)) if el not in la]) for la in presence[j]]
            presence[j] = [la + [el] for la, el in zip(presence[j], choice)]
            seq = ''.join([ntuples[i] for i in choice])[:2000]
            b[j] = encoder(seq)
        base.append(b)
    base = np.stack(base)
    baseline_file = os.path.join(outdir, '{}-balanced-{}-{}_baseline.npy'.format(name, num_seq, n))
    np.save(baseline_file, base)
    print('Balanced baseline size {} was written into {}'.format(base.shape, baseline_file))
    return baseline_file


def produce_morebalanced_baseline(outdir, name, num_seq, n=3, same_for_each_seq=True):
    def kmers(k, sigma="ACGT"):
        if k == 1:
            for s in sigma:
                yield [s]
        else:
            kms = kmers(k - 1, sigma)
            for k in kms:
                for l in sigma:
                    yield k + [l]

    def generate_seqs(l=1, k=1, sigma="ACGT"):
        result = list(kmers(k, sigma))
        sigma = list(sigma)
        for i in range(max(0, l - k)):
            # extend the lists by one
            random.shuffle(sigma)
            result.sort()
            for i, r in enumerate(result):
                r.insert(0, sigma[i % len(sigma)])
        return ["".join(r) for r in result]

    trials = 4 ** n
    encoder = OHEncoder()
    d = encoder.dictionary
    if same_for_each_seq:
        ss = generate_seqs(l=2000, k=n, sigma=d)
        ss.sort()
        b = np.stack([np.array(encoder(seq)) for seq in ss], axis=0)
        base = np.stack([b for _ in range(num_seq)], axis=1)
    else:
        base = np.zeros((trials, num_seq, 4, 2000))
        for num in range(num_seq):
            ss = generate_seqs(l=2000, k=n, sigma=d)
            ss.sort()
            b = np.stack([np.array(encoder(seq)) for seq in ss], axis=0)
            base[:, num, :, :] = b
    baseline_file = os.path.join(outdir, '{}-morebalanced-{}-{}_baseline.npy'.format(name, num_seq, n))
    np.save(baseline_file, base)
    print('Balanced baseline size {} was written into {}'.format(base.shape, baseline_file))
    return baseline_file


def produce_patient_specific_baseline_and_query(promoter_id,
                                                ref_seq='patient_specific_thresh2_ref_subset.fasta',
                                                pos_seq='patient_specific_thresh2_pos_subset.fasta',
                                                neg_seq='patient_specific_thresh2_neg_subset.fasta',
                                                working_dir='/home/marni/magisterka/data/patient_specific/'):
    print('Patient-specific baseline and query matrix for promoter {}'.format(promoter_id))
    base_seqs = []
    for record in SeqIO.parse(os.path.join(working_dir, ref_seq), "fasta"):
        if promoter_id in record.description:
            base_seqs.append(str(record.seq))
            break
    for record in SeqIO.parse(os.path.join(working_dir, pos_seq), "fasta"):
        if promoter_id in record.description:
            base_seqs.append(str(record.seq))
    n = len(base_seqs)
    print('{} sequences added to the baseline (from ref and pos files)'.format(n))

    query_seqs = []
    for record in SeqIO.parse(os.path.join(working_dir, neg_seq), 'fasta'):
        if promoter_id in record.description:
            query_seqs.append(record)
    num_seq = len(query_seqs)
    print('{} sequences added to the query (from neg file)'.format(num_seq))

    encoder = OHEncoder()
    b = np.stack([np.array(encoder(seq.upper())) for seq in base_seqs], axis=0)
    base = np.stack([b for _ in range(num_seq)], axis=1)
    baseline_file = os.path.join(working_dir, '{}-patient-specific-{}-{}_baseline.npy'.format(promoter_id, num_seq, n))
    np.save(baseline_file, base)
    print('Patient-specific baseline for {}, size {}, was written into {}'.format(promoter_id, base.shape, baseline_file))

    query_file = os.path.join(working_dir, 'patient-specific_{}_neg.fasta'.format(promoter_id))
    with open(query_file, 'w') as f:
        for seq in query_seqs:
            f.write('>{}\n{}\n'.format(seq.description, seq.seq))
    print('Patient-specific query for {}, num seqs {}, was written into {}'.format(promoter_id, len(query_seqs), query_file))

    return baseline_file, query_file


def patient_specific_extreme_seqs(min_pos, min_neg,
                                  ref_seq='patient_specific_thresh2_ref_subset.fasta',
                                  pos_seq='patient_specific_thresh2_pos_subset.fasta',
                                  neg_seq='patient_specific_thresh2_neg_subset.fasta',
                                  working_dir='/home/marni/magisterka/data/patient_specific/'):
    d = {}
    for record in SeqIO.parse(os.path.join(working_dir, ref_seq), "fasta"):
        idd = record.description.strip().split(' ')[5]
        assert idd.startswith('E') and idd not in d
        d[idd] = [0, 0]

    for record in SeqIO.parse(os.path.join(working_dir, pos_seq), "fasta"):
        idd = record.description.strip().split(' ')[5]
        assert idd.startswith('E') and idd in d
        d[idd][0] += 1

    for record in SeqIO.parse(os.path.join(working_dir, neg_seq), "fasta"):
        idd = record.description.strip().split(' ')[5]
        assert idd.startswith('E') and idd in d
        d[idd][1] += 1

    selected_ids = []
    for name, (pos, neg) in d.items():
        if pos >= min_pos and neg >= min_neg:
            selected_ids.append(name)
            print('Promoter {} selected: pos {}, neg {}'.format(name, pos, neg))
    print('{} sequences selected'.format(len(selected_ids)))

    created_sets = {}
    for seq in selected_ids:
        baseline_file, query_file = produce_patient_specific_baseline_and_query(seq,
                                                                                ref_seq=ref_seq,
                                                                                pos_seq=pos_seq,
                                                                                neg_seq=neg_seq,
                                                                                working_dir=working_dir)
        created_sets[seq] = [baseline_file, query_file]

    return created_sets
