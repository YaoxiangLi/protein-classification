import argparse
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
import utils

PATH = "data/"

def main():
    '''
    Main function: set the parameters & call training.
    Training parameters can be adjusted here.
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input_file', type=str, default="data/chloroplast.fasta",
                        help='The address to the input fasta file.')
    parser.add_argument('-output_file', type=str, default="data/protein_samples.fa",
                        help='The address to write the generated fasta sequences.')
    parser.add_argument('-batch_size', type=int, default=50,
                        help='Minibatch size.')
    parser.add_argument('-num_layers', type=int, default=3,
                        help='Number of layers in the RNN.')
    parser.add_argument('-hidden_dim', type=int, default=100,
                        help='Size of RNN hidden state.')
    parser.add_argument('-seq_length', type=int, default=100,
                        help='RNN sequence length.')
    parser.add_argument('-train_epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('-generate_epochs', type=int, default=50,
                        help='Number of generating epochs.')

    args = parser.parse_args()
    train(args)


def train(**kwargs):
    orig_seqs = []
    with open(kwargs.get('input_file'), 'r') as infasta:
        for _, seq in read_fasta(infasta):
            orig_seqs.append(seq)
    data_x, data_y, vocab_size, vocab_decode = \
        load_data(orig_seqs, seq_length)
    rnn_model = Sequential()
    rnn_model.add(LSTM(hidden_dim,
                       input_shape=(None, vocab_size),
                       return_sequences=True))
    for i in range(num_layers - 1):
        rnn_model.add(LSTM(hidden_dim,
                           return_sequences=True))
    rnn_model.add(TimeDistributed(Dense(vocab_size)))
    rnn_model.add(Activation('softmax'))
    rnn_model.compile(loss="categorical_crossentropy",
                      optimizer="rmsprop")
    rnn_model.fit(data_x,
                  data_y,
                  batch_size=batch_size,
                  verbose=1,
                  epochs=train_epochs)




if __name__ == '__main__':
    main()
    kwargs = {
        'input_file': 'data/chloroplast.fasta',
        'output_file': 'data/test.fasta',
        'batch_size': 50,
        'num_layers': 3,
        'hidden_dim': 100,
        'seq_length': 100,
        'train_epochs': 100,
        'generate_epochs': 50,
    }
