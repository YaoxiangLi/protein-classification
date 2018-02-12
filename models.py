import tensorflow as tf


def rnn_model(data, args):
    with tf.variable_scope("rnn",
                           initializer=\
                           tf.contrib.layers.variance_scaling_initializer()):
        if args.rnn_type == 'basic':
            basic_cells = [tf.contrib.rnn.BasicLSTMCell(
                           num_units=args.hidden_dim)
                           for layer in range(args.n_layers)]
            drop_cells = [tf.contrib.rnn.DropoutWrapper(cell,
                          input_keep_prob=args.in_keep_prob)
                          for cell in basic_cells]
            multi_cells = tf.contrib.rnn.MultiRNNCell(drop_cells)
            _, states = tf.nn.dynamic_rnn(multi_cells, data, dtype=tf.float32)
            final_state = states[-1][1]
        elif args.rnn_type == 'lstm':
            lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=args.hidden_dim)
                          for layer in range(args.n_layers)]
            drop_cells = [tf.contrib.rnn.DropoutWrapper(cell,
                          input_keep_prob=args.in_keep_prob)
                          for cell in lstm_cells]
            multi_cells = tf.contrib.rnn.MultiRNNCell(drop_cells)
            _, states = tf.nn.dynamic_rnn(multi_cells, data, dtype=tf.float32)

            final_state = states[-1][1]
        elif args.rnn_type == 'gru':
            gru_cells = [tf.contrib.rnn.GRUCell(num_units=args.hidden_dim)
                         for layer in range(args.n_layers)]
            drop_cells = [tf.contrib.rnn.DropoutWrapper(cell,
                          input_keep_prob=args.in_keep_prob)
                          for cell in gru_cells]
            multi_cells = tf.contrib.rnn.MultiRNNCell(drop_cells)
            _, states = tf.nn.dynamic_rnn(multi_cells, data, dtype=tf.float32)

            final_state = states[-1]
        else:
            raise Exception("\ns{} is not a type!".format(args.rnn_type))

    return final_state
