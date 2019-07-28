import numpy as np
import tensorflow as tf
import math

from dataset import MnistDataset

IMG_SIZE = 28


class CNNMnistLayer:
    def __init__(self, filters: list, kernel_size: int = 3, name: str = None):
        self.layers = []
        self.name = name
        for index, filter_count in enumerate(filters):
            self.layers.append(tf.compat.v1.layers.Conv2D(filters=filter_count,
                                                          kernel_size=kernel_size, name=f'conv2d_{index}',
                                                          input_shape=(IMG_SIZE, IMG_SIZE, 1),
                                                          padding='same',
                                                          activation=tf.nn.relu))
            self.layers.append(tf.compat.v1.layers.MaxPooling2D(pool_size=[2, 2], strides=2,
                                                                name=f'maxpool2d_{index}'))

    def __call__(self, input_tensor: tf.Tensor):
        with tf.name_scope(self.name):
            for layer in self.layers:
                output = layer(input_tensor)
                input_tensor = output
        return output


class Network:
    def __init__(self):
        self.logits = None

    def get_network(self):
        X = tf.placeholder(tf.float32, shape=(None, IMG_SIZE, IMG_SIZE, 1), name='input')

        cnn_layers_0 = CNNMnistLayer([32, 64], kernel_size=3, name='cnn_layer_0')
        # batch_norm_layer = tf.layers.BatchNormalization()
        # cnn_layers_1 = CNNMnistLayer([64], kernel_size=3, name='cnn_layer_1')

        dense = tf.compat.v1.layers.Dense(1024, activation=tf.nn.relu)
        logits = tf.compat.v1.layers.Dense(10, activation=tf.nn.softmax, name='logits')

        mnist_nn = cnn_layers_0(X)
        # mnist_nn = cnn_layers_1(mnist_nn)
        # mnist_nn = tf.reshape(mnist_nn, [-1, 3 * 3 * 64])
        mnist_nn = tf.reshape(mnist_nn, [-1, 7 * 7 * 64])

        mnist_nn = dense(mnist_nn)
        self.logits = logits(mnist_nn)

        return self.logits


class Train:
    def __init__(self, network: tf.Tensor):
        self.network = network
        self.mnist = MnistDataset()
        self.loss = None

    def train(self):

        train_number = len(self.mnist.train_labels)
        batch_size = 1000
        batches_number = math.ceil(train_number / batch_size)
        epochs_number = 20

        with tf.Session() as sess:
            sess.run(tf.compat.v1.initializers.global_variables())
            tf.summary.FileWriter('./tb_logs', sess.graph)

            # deal with epochs
            for n_epoch in range(epochs_number):
                # deal with batches
                for batch_number in range(batches_number):
                    labels = np.eye(10)[self.mnist.train_labels[batch_number*batch_size:(batch_number+1)*batch_size]]
                    self.loss = tf.compat.v1.losses.mean_squared_error(labels, self.network)
                    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

                    images = self.mnist.train_images[batch_number*batch_size:(batch_number+1)*batch_size]
                    _, loss_val = sess.run([optimizer, self.loss], feed_dict={'input:0': images})
                    print(f'Epoch number: {n_epoch}, batch number: {batch_number}: loss: {loss_val}')

                print(f'Loss after epoch {n_epoch}: {loss_val}')
