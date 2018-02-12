import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data_csv(infile):
    data = pd.read_csv(infile, header=0, sep=',')
    char_set = list(set("".join(data['sequence'])))
    vocab_size = len(char_set)
    vocab = dict(zip(char_set, range(vocab_size)))
    # Split data into train & test
    train_df, test_df = train_test_split(data, test_size=0.1)

    # Embedding the characters using the dictioary built above. Basically,
    # replacing  each character with its numerical code in the dictionary.
    train_seqs = np.array([list(map(vocab.get, k))
                          for k in train_df['sequence']])
    test_seqs = np.array([list(map(vocab.get, k))
                          for k in test_df['sequence']])

    # Converting categorical labels to actual numbers For two categories of
    # labels, for example, we'll have classes [0, 1].
    train_labels = np.array(train_df['label'].astype('category').cat.codes)
    test_labels = np.array(test_df['label'].astype('category').cat.codes)

    return train_seqs, test_seqs, train_labels, test_labels


def next_batch(x_data, y_data, batch_size):
    '''
    Returns batches of x & y
    '''
    # Choose a random set of row indices with the size of batch_size
    idx = np.random.choice(np.arange(len(x_data)),
                           size=batch_size,
                           replace=False)
    # Return the subset (batch) of data using the randomly chosen row indices
    return x_data[idx, :], y_data[idx]


def load_data(orig_seqs, seq_length):
    orig_seqs = " ".join(orig_seqs)
    chars = list(set(orig_seqs))
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))
    vocab_decode = dict(zip(range(vocab_size), chars))

    # one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    seqs_x = np.zeros((len(orig_seqs) // seq_length, seq_length, vocab_size))
    seqs_y = np.zeros((len(orig_seqs) // seq_length, seq_length, vocab_size))

    for i in range(len(orig_seqs) // seq_length):
        # one-hot encoding
        embed_x = [vocab_embed[v]
                   for v in orig_seqs[i * seq_length:(i + 1) * seq_length]]
        seqs_x[i, :, :] = np.array([vocab_one_hot[j, :] for j in embed_x])

        embed_y = [vocab_embed[v]
                   for v in orig_seqs[i * seq_length +
                   1:(i + 1) * seq_length + 1]]
        seqs_y[i, :, :] = np.array([vocab_one_hot[j, :] for j in embed_y])

    return seqs_x, seqs_y, vocab_size, vocab_decode


def read_fasta(infasta):
    name, seq = None, []
    for line in infasta:
        line = line.rstrip()
        if line.startswith(">"):
            if name:
                yield(name.replace(">", ""), ''.join(seq))
            name, seq = line, []
        else:
            seq.append(line)
    if name:
        yield(name.replace(">", ""), ''.join(seq))
