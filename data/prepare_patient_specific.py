import pandas as pd
from statistics import mean, stdev
import os
from Bio.Seq import Seq
from Bio import SeqIO
import math

chr_order = ['chr{}'.format(i) for i in range(1, 23)] + ['chrX', 'chrY']
path = '/home/marni/magisterka/data/patient_specific'
common_path = '/home/marni/magisterka/data'
thresh = 2
nucleotides = ['A', 'C', 'T', 'G', 'N']

coverage = pd.read_csv(os.path.join(path, 'H3K27ac_coverage_normalised_on_promoters.tsv'), sep='\t', index_col=0)
activity = pd.read_csv(os.path.join(common_path, 'H3K27ac_active_transcripts_table_with_activity_in_16_patients.csv'),
                       sep='\t', index_col=0)

activity = activity[
    ((activity['sum'] >= 1) & (activity['sum'] <= 3)) | ((activity['sum'] >= 13) & (activity['sum'] <= 15))]
coverage = coverage.loc[activity.index][activity.columns[:-1]]

accepted_file = os.path.join(path, 'accepted_thresh{}.txt'.format(thresh))
if os.path.isfile(accepted_file):
    accepted = open(os.path.join(path, 'accepted_thresh{}.txt'.format(thresh)), 'r').read().strip().split('\n')
else:
    accepted = []
    for i, row in coverage.iterrows():
        activity_row = activity.loc[i]
        if activity_row['sum'] >= 13:
            neg_value = 0
        elif activity_row['sum'] <= 3:
            neg_value = 1
        activity_row.drop('sum', inplace=True)
        neg = activity_row[activity_row == neg_value].index
        pos = [el for el in activity_row.index if el not in neg]
        m = mean(row[pos])
        s = stdev(row[pos])
        zscore = [(row[el] - m) / s if s != 0 else 0 for el in neg]
        if all([abs(el) > thresh for el in zscore]):
            accepted.append(i)
            print('{} accepted'.format(i))
    with open('accepted_thresh{}.txt'.format(thresh), 'w') as w:
        w.write('\n'.join(accepted))
    del row, activity_row, neg_value, neg, pos, m, s, zscore

del coverage
print('{} accepted promoters'.format(len(accepted)))

loc = pd.read_csv(os.path.join(common_path, 'TSSs_1000_flanked_transcripts_ids_genes_ids.bed'), header=None, sep='\t')
loc.set_index(3, inplace=True)
loc.columns = ['chr', 'start', 'end', 'idk', 'strand', 'gene']
loc.drop('idk', axis=1, inplace=True)
loc = loc.loc[accepted]
loc = loc[~loc.duplicated(['chr', 'start'], keep='first')]

snps = pd.read_csv(os.path.join(path, 'all_panel_SNPs.csv'), sep='\t', index_col=0)
snps.columns = [el.replace('.GT', '') for el in snps.columns]
snps = snps[['POS', 'REF', 'ALT'] + list(activity.columns[:-1])]

wref = open(os.path.join(path, 'patient_specific_thresh{}_ref_test.fasta'.format(thresh)), 'w', 1)
wpos = open(os.path.join(path, 'patient_specific_thresh{}_pos_test.fasta'.format(thresh)), 'w', 1)
wneg = open(os.path.join(path, 'patient_specific_thresh{}_neg_test.fasta'.format(thresh)), 'w', 1)
wrong_ref = 0
for ch in [el for el in chr_order if el in loc['chr'].unique()]:
    print(ch)
    ref_file = os.path.join(common_path, 'hg38/{}.fasta'.format(ch))
    for record in SeqIO.parse(ref_file, "fasta"):
        ref = record.seq
        break
    i, j, num_alts = 0, [], []
    for ind, row in loc[loc['chr'] == ch].iterrows():
        seq = ref[row['start']:row['end'] - 1]
        if len(seq) != 2000:
            print('Wrong seq length: chr {}, clip {}, start: {}, end: {}'.format(len(ref), len(seq), \
                                                                                 row['start'], row['end'] - 1))
            raise ValueError
        midpoint = row['start'] + math.ceil((row['end'] - row['start']) / 2)
        snps_clip = snps.loc[ch]
        snps_clip = snps_clip[(snps_clip['POS'] - 1 >= row['start']) & (snps_clip['POS'] - 1 < row['end'] - 1)]

        activity_row = activity.loc[ind]
        if activity_row['sum'] >= 13:
            ref_class = 'promoter active'
            ref_value = 1
            alt_class = 'promoter inactive'
            alt_value = 0
        elif activity_row['sum'] <= 3:
            ref_class = 'promoter inactive'
            ref_value = 0
            alt_class = 'promoter active'
            alt_value = 1
        else:
            raise ValueError
        activity_row.drop('sum', inplace=True)
        neg = activity_row[activity_row == alt_value].index
        pos = activity_row[activity_row == ref_value].index
        assert len(neg) + len(pos) == 16

        # write reference sequence
        if row['strand'] == '-':
            wref.write('> {} {} {} {} {} REF\n{}\n'.format(ch, midpoint, row['strand'], ref_class, ind,
                                                           seq.reverse_complement()))
        else:
            wref.write('> {} {} {} {} {} REF\n{}\n'.format(ch, midpoint, row['strand'], ref_class, ind, seq))
        alt_versions, num_snps_pats = set([]), set([])
        for pat in list(pos) + list(neg):
            new_seq = str(seq)
            num_snps = 0
            positions = set()
            for snp in [el for _, el in snps_clip.iterrows()
                        if not all([la in [el['REF'], '.', '*'] for la in el[pat].split('/')])]:
                pos_rel = snp['POS'] - 1 - row['start']
                assert pos_rel not in positions, positions
                if snp[pat][0] not in nucleotides or snp[pat][0] == snp['REF']:
                    alt_bp = snp[pat][-1]
                    if alt_bp not in nucleotides or alt_bp == snp['REF']:
                        continue
                else:
                    alt_bp = snp[pat][0]
                if new_seq[pos_rel].isupper():
                    alt_bp = alt_bp.upper()
                else:
                    alt_bp = alt_bp.lower()
                if new_seq[pos_rel].upper() != snp['REF']:
                    wrong_ref += 1
                    print('Wrong reference SNP for {}, pat {}, position {}, REF {}, true seq(+-2): {}'. \
                          format(ch, pat, snp['POS'], snp['REF'], str(s)[pos_rel - 2:pos_rel + 3]))
                new_seq = new_seq[:pos_rel] + alt_bp + new_seq[pos_rel + 1:]
                num_snps += 1
                positions.add(pos_rel)
            if num_snps > 0:
                j.append(num_snps)
                alt_versions.add(new_seq)
                num_snps_pats.add(num_snps)
                if row['strand'] == '-':
                    new_seq = str(Seq(new_seq).reverse_complement())
                if pat in pos:
                    wpos.write('> {} {} {} {} {} {} {}SNPs\n{}\n'.format(ch, midpoint, row['strand'], ref_class,
                                                                         ind, pat, num_snps, new_seq))
                else:
                    wneg.write('> {} {} {} {} {} {} {}SNPs\n{}\n'.format(ch, midpoint, row['strand'], alt_class,
                                                                         ind, pat, num_snps, new_seq))
                i += 1
        if len(alt_versions) > 1:
            num_alts.append(len(alt_versions))
            for alt_version in alt_versions:
                nn = sum(1 for a, b in zip(alt_version, seq) if a != b)
                assert nn in num_snps_pats

    del ref, record, ref_file, ind, row, seq, midpoint, snps_clip, activity_row, neg, pos, pat, new_seq, num_snps, \
        positions, alt_versions, num_snps_pats
    if not j:
        j = [0]
    if not num_alts:
        num_alts = [0]
    print(i, mean(j), mean(num_alts))
wref.close()
wneg.close()
wpos.close()
