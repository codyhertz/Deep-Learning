import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# download dataset
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# nodes per layer, 3 layers
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10 # (0-9)
batch_size = 50 # feeds 50 features at a time, makes sure memory isnt exceeded by huge datasets

# gives shape of data
x = tf.placeholder('float', [None, 784]) # 784 floats in a tensor
y = tf.placeholder('float')


def neural_network_model(data):
    # create dictionary of tensors for hidden layers and output layer in network
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases']) # layer 1 output
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases']) # layer 2 output
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases']) # layer 3 output
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases'] # output of network

    return output


def train_neural_network(x):
    prediction = neural_network_model(x) # create graph of network
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)) # cost function of network
    optimizer = tf.train.AdamOptimizer().minimize(cost) # function for minimizing the output of the cost function

    how_many_epochs = 35 # how many cycles of training the network will go through

    # begin training the model
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables()) # create all of the tensor objects

        # forward and backward propogate for the number of epochs
        for epoch in range(how_many_epochs):
            epoch_loss = 0 # variable to track the loss of the current epoch

            # iterate through each batch
            for i in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size) # get training data and labels for the current epoch
                i, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y}) # do propogation
                epoch_loss += c # keep track of loss per epoch to show progress of the learning

            print('Epoch', epoch, 'completed out of', how_many_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)) # compare predicted value to the test values
        accuracy = tf.reduce_mean(tf.cast(correct, 'float')) # get accuracy based off of correct
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x)
