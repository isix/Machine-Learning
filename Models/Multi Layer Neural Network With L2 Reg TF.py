# Multi Layer Neural Network With L2 Regularization Using Tensorflow
# https://markojerkic.com/build-a-multi-layer-neural-network-with-l2-regularization-with-tensorflow/

import tensorflow as tf
import pandas as pd
import numpy as np

filename = "mushroom.csv"

'''''
true_class: edible=0,poisonous=1
neg_class: poisonous=0, edible=1
cap-shape: bell=0,conical=1,convex=2,flat=3, knobbed=4,sunken=5
cap-surface: fibrous=0,grooves=1,scaly=2,smooth=3
cap-color: brown=0,buff=1,cinnamon=2,gray=3,green=4,pink=5,
    purple=6,red=7,white=8,yellow=9
odor: almond=0,anise=1,creosote=2,fishy=3,foul=4,musty=5,
    none=6,pungent=7,spicy=8
'''''

# Read the csv file into a DataFrame object
data = pd.read_csv(filename)

# Convert DataFrame object into a numpy npdarray
data = data.as_matrix()

# Define sizes of training and validations sets
train_size = 3000
valid_size = 1000
# Number of columns to use
num_fields = 4

# Choose a random subset
indices = np.random.choice(8000, train_size)
train_data = data[indices, :]

indices = np.random.choice(8000, valid_size)
valid_data = data[indices, :]
# Seperate data from labels
train_dataset = train_data[:, 2:]
train_labels = train_data[:, :2]

valid_dataset = valid_data[:, 2:]
valid_labels = valid_data[:, :2]
# Clear space in memory
del data
del valid_data
del train_data

# Define size of hidden layers
num_nodes_1 = 1024
num_nodes_2 = int(num_nodes_1 * 0.5)
num_nodes_3 = int(num_nodes_1 * np.power(0.5, 2))
# Number of possible outputs
num_labels = 2
# Batch size
batch_size = 128
# Beta for L2 regularization
beta = 0.01

# Create a Tensorflow graph
graph = tf.Graph()

with graph.as_default():
    # Create weights and biases
    def weights_and_biases(a, b):
        w = tf.Variable(tf.truncated_normal(shape=[a, b], stddev=np.sqrt(2 / a)))
        b = tf.Variable(tf.zeros([b]))
        return w, b

    # Create tensors for training data and labels and for
    # validation data
    tf_train_data = tf.placeholder(tf.float32, shape=[batch_size, num_fields])
    tf_train_labels = tf.placeholder(tf.float32, shape=[batch_size, num_labels])
    tf_valid_data = tf.constant(valid_dataset, dtype=tf.float32)

    # Assign weights and biases
    weights_1, biases_1 = weights_and_biases(num_fields, num_nodes_1)
    weights_2, biases_2 = weights_and_biases(num_nodes_1, num_nodes_2)
    weights_3, biases_3 = weights_and_biases(num_nodes_2, num_nodes_3)
    weights_4, biases_4 = weights_and_biases(num_nodes_3, num_labels)

    # Compute relu logits
    def relu_logits(data, drop=False):
        logits = tf.nn.relu(tf.matmul(data, weights_1) + biases_1)
        # Are we going to drop some values
        if drop:
            logits = tf.nn.dropout(logits, 0.5)
        logits = tf.nn.relu(tf.matmul(logits, weights_2) + biases_2)
        if drop:
            logits = tf.nn.dropout(logits, 0.5)
        logits = tf.nn.relu(tf.matmul(logits, weights_3) + biases_3)
        if drop:
            logits = tf.nn.dropout(logits, 0.5)
        logits = tf.matmul(logits, weights_4) + biases_4

        return logits

    # Logits for training data
    logits = relu_logits(tf_train_data, drop=True)

    # Regular loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=tf_train_labels))
    # L2 loss
    reg = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(weights_3) + \
          tf.nn.l2_loss(weights_4)
    loss = tf.reduce_mean(loss + reg * beta)

    # Optimizer
    start_learn_rate = 0.1
    global_step = tf.Variable(0)
    # Create a decaying learning rate
    # start, global step, decay step, decay rate, staircase
    learn_rate = tf.train.exponential_decay(start_learn_rate, global_step, 100000, 0.5, staircase=True)
    optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    # Training predictions
    train_predictions = tf.nn.softmax(logits)

    # Validation predictions
    logits = relu_logits(tf_valid_data)
    valid_predictions = tf.nn.softmax(logits)

# Number of training steps
num_steps = 5001

# Compute accuracy of predictions if %
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

# Create a Tensorflow session
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        # Deffine an offset
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
         # Create batch data and labels
        batch_data = train_dataset[offset:(offset + batch_size)]
        batch_labels = train_labels[offset:(offset + batch_size)]

        feed_dict = {tf_train_data: batch_data, tf_train_labels: batch_labels}

        _, l, predictions = session.run([optimizer, loss, train_predictions], feed_dict=feed_dict)

        if step % 1000 == 0:
            print("Minibatch loss at step {}: {}".format(step, l))
            print("Minibatch accuracy: {:.1f}".format(accuracy(predictions, batch_labels)))
            print("Validation accuracy: {:.1f}".format(accuracy(valid_predictions.eval(),
                                                                valid_labels)))
