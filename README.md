### Using convolution neural networks (CNN) for better understanding of human genome

<br></br>

**This repository is a part of my master's project called "Identification of 
chromatin regions active in human brain using neural networks".**

<br></br>

### Versions

- Python 3.6
- pandas 1.0.1
- numpy 1.18.1
- pytorch 1.3.1

<br></br>

### Example dataset

Example of trained networks are in */data/custom40* and */data/alt1* directories.


**Training an existing network on new samples**

To re-train a network on newly available data, you can use the *train_new.py* script.

The simplest way to invoke is as follows:

```
 python3 train_new.py 

```

It will then run an analysis based on the data in the *data/test_fasta* and the model from *data/alt1* directory.

If you this runs properly you can start your own analysis by changing some parameters. Let us start by looking at the options:

```

bartek@nu:~/Dropbox/code/RegSeqNet$ python3 train_new.py --help
usage: train_new.py [-h] [--model NAME] [-p DIR] [-x PREFIX] [-o DIR] [--seed NUMBER] [--optimizer NAME] [--loss_fn NAME] [--batch_size INT] [--num_workers INT] [--num_epochs INT]
                    [--acc_threshold FLOAT] [--learning_rate FLOAT] [--no_adjust_lr] [--seq_len INT] [--dropout-conv FLOAT] [--dropout-fc FLOAT] [--weight-decay FLOAT] [--momentum FLOAT]

Train network based on given data

options:
 -h, --help            show this help message and exit
  --model NAME          File with the model weights to load before training
  --namespace NAME      The namespace for this run of training
  -p DIR, --path DIR    Working directory.
  -x PREFIX, --prefix PREFIX
                        file_prefix.
  -o DIR, --output DIR  Output directory, default: test_output
  --seed NUMBER         Set random seed, default: 0
  --optimizer NAME      optimization algorithm to use for training the network, default = RMSprop
  --loss_fn NAME        loss function for training the network, default = CrossEntropyLoss
  --batch_size INT      size of the batch, default: 64
  --num_workers INT     how many subprocesses to use for data loading, default: 4
  --num_epochs INT      maximum number of epochs to run, default: 300
  --acc_threshold FLOAT
                        threshold of the validation accuracy - if gained training process stops, default: 0.9
  --learning_rate FLOAT
                        initial learning rate, default: 0.01
  --no_adjust_lr        no reduction of learning rate during training, default: False
  --seq_len INT         Length of the input sequences to the network, default: 2000
  --dropout-conv FLOAT  Dropout of convolutional layers, default value is 0.2
  --dropout-fc FLOAT    Dropout of fully-connected layers, default value is 0.5
  --weight-decay FLOAT  Weight decay, default value is 0.0001
  --momentum FLOAT      Momentum, default value is 0.1
```

If we just want to use a different fasta files, without modifying the training params, you can simply run:

```
python3 --path data/my_fasta --prefix fasta_file -output data/my_output --namespace MY-RUN-1
```

for this we assume that you have 4 fasta files in the *data/my_fasta* folder and that the *data/my_output* folder is created.



**Calculating the outputs for a full fasta file**


**Calculating integrated gradients**
<br></br>
To calculate integrads based on example model and set of sequences just run:

```
python3 calculate_integrads.py \
        --model custom40_last.model \
        --seq extreme_custom40_train_1.fasta \
        --baseline CHOSEN-BASELINE
```
CHOSEN-BASELINE depends on what baseline you want to use for calculating 
integrated gradients (see: https://arxiv.org/abs/1703.01365 for details 
of the method), select one of the options:
- *zeros* - use zeros array as baseline
- *fixed* - use the same set of random sequences as the baseline for each 
sequence
- *random* - use different random set of sequences as the baseline for each 
sequence
- *test-balanced-8-3_baseline.npy* - use pre-calculated balanced baseline 
(each 3 nucleotides in the given position occur exactly once across all 64. baseline sequences)

As the output new directory called *integrads_NETWORK_SEQUENCES_BASELINE_TRIALS-STEPS* is created
(*integrads_custom40_extreme-custom40-train-1_CHOSEN-BASELINE_10-50* if the default data was used). 
Inside there are result files:
- integrads_all.npy - numpy array with calculated gradients
- params.txt - file with parameters of the analysis

**Plotting seqlogos**
<br></br>
To plot seqlogos based on the calculated integrads run:
```
python3 plot_seqlogo.py \
integrads_custom40_extreme-custom40-train-1_CHOSEN-BASELINE_10-50/integrads_all.npy \
--global_ylim \
--one \
--clip NUM
```
Options *global_ylim*, *one* and *clip* are optional:
- *global_ylim* - 
set the same y axis range for all sequences from the given array
- *one* - plot only one nucleotide in one position 
(the one from the original sequence)
- *clip* - subset of nucleotides to plot: +-NUM from the middle 
of the sequences, by default NUM=100

As the output new directory with plots is created.


