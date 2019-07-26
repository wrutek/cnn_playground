import numpy as np
import tensorflow as tf

from network import Network, IMG_SIZE

if __name__ == '__main__':

    graph = tf.Graph()
    tf.reset_default_graph()
    with graph.as_default():

        net = Network()
        mnist_nn = net.get_network()

        input_arr = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3)

        with tf.Session() as sess:
            sess.run(tf.initializers.global_variables())
            pred = sess.run(mnist_nn, feed_dict={'input:0': input_arr})
            print(pred)
