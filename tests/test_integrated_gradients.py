from bin.integrated_gradients import *
from bin.datasets import *
import pytest
from bin.networks import TestNetwork

@pytest.mark.parametrize("input_dir", ['/test/data/test.fasta'])
def test_integrated_gradients(input_dir):
    dataset = SeqsDataset(os.path.join('/home/marni/PycharmProjects/magisterka', input_dir), seq_len=10)
    X, y = dataset.__getitem__(0)
    labels = [y]
    X = [X]
    for i in range(1, len(dataset)):
        xx, yy = dataset.__getitem__(i)
        X.append(xx)
        labels.append(yy)
    X = torch.stack(X, dim=0)
    model = TestNetwork(10)
    results = integrated_gradients(model, X, 0)
    #results = calculate_gradients(X, model, labels)
    print(results)
