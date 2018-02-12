import argparse
import numpy as np
import tensorflow as tf
from utils import load_data_csv, next_batch
from models import rnn_model


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input_file', type=str,
                        default='data/proteins12.csv',
                        help='The path to the input file.')
    parser.add_argument('-rnn_type', type=str, default='gru',
                        help='The type of RNN network: basic, lstm, gru.')
    parser.add_argument('-batch_size', type=int, default=64,
                        help='Minibatch size.')
    parser.add_argument('-n_layers', type=int, default=2,
                        help='Number of layers in the RNN.')
    parser.add_argument('-hidden_dim', type=int, default=2000,
                        help='Size of RNN hidden state.')
    parser.add_argument('-n_inputs', type=int, default=401,
                        help='Protein sequence length.')
    parser.add_argument('-n_classes', type=int, default=2,
                        help='Number of classes; protein families/clusters.')
    parser.add_argument('-in_keep_prob', type=float, default=1,
                        help='Dropout probability.')
    parser.add_argument('-learning_rate', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('-n_epochs', type=int, default=100,
                        help='Number of training epochs.')

    args = parser.parse_args()

    x_train, x_test, y_train, y_test = load_data_csv(args.input_file)

    classes = np.sort(np.unique(y_train))
    print("\n=================================\nData details:")
    print("- Training-set:\t\t{}".format(len(y_train)))
    print("- Test-set:\t\t{}".format(len(y_test)))
    print("- Features:\t\t{}".format(args.n_inputs))
    print("- Classes:\t\t{}".format(classes))
    print("=================================\n\n")

    train(x_train, x_test, y_train, y_test, args)


def train(x_train, x_test, y_train, y_test, args):
    x_input = tf.placeholder(tf.float32, [None, 1, args.n_inputs])
    target = tf.placeholder(tf.int32, [None])
    n_test_examples = len(y_test)
    x_test = x_test.reshape((n_test_examples, 1, args.n_inputs))
    final_state = rnn_model(x_input, args)

    # Calculate probabilities
    with tf.name_scope('Logits'):
        logits = tf.layers.dense(final_state, args.n_classes)

    # Loss function
    with tf.name_scope('Loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits)
        loss = tf.reduce_mean(xentropy)

    # Optimization
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        training_op = optimizer.minimize(loss)

    # Accuracy
    with tf.name_scope('Accuracy'):
        # Whether the targets are in the top K predictions
        correct = tf.nn.in_top_k(logits, target, k=1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # To initialize all TF variables
    init = tf.global_variables_initializer()

    best_accuracy = 0.0
    last_improvement = 0
    check_improvement = 100
    n_train_examples = len(y_train)

    # The graph
    with tf.Session() as sess:
        init.run()
        for epoch in range(args.n_epochs):
            for i in range(n_train_examples // args.batch_size):
                # Get the trainig batch
                X_batch, y_batch = next_batch(x_train, y_train,
                                              args.batch_size)
                # Reshape training batch, in the shape of the placeholder
                X_batch = X_batch.reshape((args.batch_size, 1, args.n_inputs))
                sess.run(training_op, feed_dict={x_input: X_batch,
                                                 target: y_batch})

            # Obtain accuracy on training & test data
            acc_train = accuracy.eval(feed_dict={x_input: X_batch,
                                                 target: y_batch})
            acc_test = accuracy.eval(feed_dict={x_input: x_test,
                                                target: y_test})
            # Update the best accuracy & output accuracy results
            if acc_test > best_accuracy:
                best_accuracy = acc_test
                last_improvement = epoch

            print("Epoch",
                  epoch,
                  "Train accuracy:",
                  acc_train,
                  "Test accuracy:",
                  acc_test)

            # Early stopping in case of no improvement.
            if epoch - last_improvement > check_improvement:
                print("No improvement seen in ",
                      check_improvement,
                      " epochs ==> early stopping")
                break


if __name__ == '__main__':
    main()
