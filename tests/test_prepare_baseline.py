import pytest
from bin.prepare_baseline import *
from bin.datasets import *


@pytest.mark.parametrize("num_seq, n", [(80, 3)])
def test_produce_balanced_baseline(num_seq, n):
    outdir = '/home/marni/Dokumenty/magisterka/'
    f = produce_balanced_baseline(outdir, 'test', num_seq, n=n)
    array = np.load(f)
    print(array.shape)
    encoder = OHEncoder()
    dif = [set([]) for _ in range(array.shape[1])]
    for base in array:
        dif = [dif[i] | {encoder.decode(el[:, :3])} for i, el in enumerate(base)]
    for el in dif:
        assert len(el) == 4**n


@pytest.mark.parametrize("num_seq, n", [(80, 3)])
def test_produce_morebalanced_baseline(num_seq, n):
    outdir = '/home/marni/magisterka/'
    f = produce_morebalanced_baseline(outdir, 'test', num_seq, n=n)
    array = np.load(f)
    print(array.shape)


def test_patient_specific_extreme_seqs():
    created_sets = patient_specific_extreme_seqs(3, 3)
    for name, files in created_sets.items():
        command = 'python calculate_integrads.py {} --path /home/marni/magisterka ' \
                  '--model patient_specific_thresh2_40000_last.model --baseline {} ' \
                  '--namespace patient_specific_thresh2_40000'.format(files[1], files[0])
        print('Command for {}\n{}'.format(name, command))

