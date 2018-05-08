
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import grid_rnn
from tensorflow.contrib import rnn

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 1

n_inputs = 28   # MNIST data input (img shape: 28*28)
n_steps = 28    # time steps
n_hidden_units = 10   # neurons in hidden layer
n_classes = 10      # MNIST classes (0-9 digits)
num_layers=2

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([1, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases,model='grid3lstm'):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> ( 28 steps, 28 inputs ,1)
    X = tf.reshape(X, [ batch_size*n_steps*n_inputs,1 ])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [n_inputs, n_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    if model == 'lstm':
        cell_fn = rnn.BasicLSTMCell
    elif model=='grid2lstm':
        cell_fn = grid_rnn.Grid2LSTMCell
    elif model=='grid3lstm':
        cell_fn=grid_rnn.Grid3LSTMCell
    else:
        raise Exception("model type not supported: {}".format(model))
    cell=cell_fn(n_hidden_units)
    # cell = rnn.MultiRNNCell([cell_fn(n_hidden_units) for _ in range(num_layers)] , state_is_tuple=True)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(n_inputs, dtype=tf.float32)

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.

    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state,dtype=tf.float32, time_major=False)
    # outputs = []
    # state = init_state
    # with tf.variable_scope("RNN"):
    #     for time_step in range(n_steps):
    #         if time_step > 0: tf.get_variable_scope().reuse_variables()
    #         (cell_output, state) = cell(X_in[:, time_step, :], state)
    #         outputs.append(cell_output)
    # # OUTPUT shape   ->(n_steps,batch_size,cell_size)
    # # after tf.stack->(batch_size,n_steps,cell_size)
    # # final output   ->(batch_size*n_steps,cell_size)
    # outputs = tf.stack(axis=1, values=outputs)
    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))


    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)
    results=tf.reshape(tf.unstack(results)[-1],[batch_size,-1])
    return results,final_state,init_state


pred,final_state,init_state = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        if step==0:
            feed_dict={
            x: batch_xs,
            y: batch_ys,
            }
        else:
            feed_dict={
            x: batch_xs,
            y: batch_ys,
            init_state: state,
            }
        _, state =sess.run([train_op,final_state], feed_dict=feed_dict)
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict=feed_dict))
        step += 1