import numpy as np
import torch
import random
import os
from bin.common import OHEncoder


def integrated_gradients(model, inputs, labels, baseline=None, num_trials=10, steps=50, use_cuda=False):
    all_integrads = []
    for i in range(num_trials):
        print('Trial {}'.format(i))
        if baseline is None:
            encoder = OHEncoder()
            base = torch.zeros(inputs.shape)
            for j, el in enumerate(inputs):
                seq = random.choices(encoder.dictionary, k=el.shape[-1])
                base[j] = torch.tensor(encoder(seq))
        elif type(baseline) is np.ndarray:
            base = torch.from_numpy(baseline[i]).reshape(inputs.shape)
        # s = (inputs - base)[0, 0]
        # ss = [el for el in s.flatten() if el != 0]
        scaled_inputs = [base + (float(j) / steps) * (inputs - base) for j in range(1, steps + 1)]
        grads = calculate_gradients(model, scaled_inputs, labels, use_cuda=use_cuda)
        avg_grads = np.average(grads[:-1], axis=0)
        integrated_grad = (inputs - base) * torch.tensor(avg_grads)
        all_integrads.append(integrated_grad)
    avg_integrads = np.average(np.stack(all_integrads), axis=0)
    return avg_integrads


def calculate_gradients(model, inputs, labels, use_cuda=False):
    torch_device = [torch.device('cuda:0') if use_cuda else torch.device('cpu')][0]
    gradients = []
    for inp, label in zip(inputs, labels):
        model.eval()
        inp = inp.float()
        inp.to(torch_device)
        inp.requires_grad = True
        output = model(inp)
        gradient = []
        for i in range(output.shape[0]):
            model.zero_grad()
            output[i][label].backward(retain_graph=True)
            gradient.append(inp.grad[i].detach().cpu().numpy())
        gradients.append(gradient)
    gradients = np.array(gradients)
    return gradients
