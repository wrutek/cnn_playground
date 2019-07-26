import tensorflow as tf

IMG_SIZE = 258


class CNNMnistLayer:
    def __init__(self, filters: list, kernel_size: int = 3, name: str = None):
        self.layers = []
        self.name = name
        for index, filter_count in enumerate(filters):
            self.layers.append(tf.layers.Conv2D(filters=filter_count,
                                                kernel_size=kernel_size, name=f'conv2d_{index}',
                                                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                padding='same',
                                                activation=tf.nn.relu))
            self.layers.append(tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2,
                                                      name=f'maxpool2d_{index}'))

    def __call__(self, input_tensor: tf.Tensor):
        with tf.name_scope(self.name):
            for layer in self.layers:
                output = layer(input_tensor)
                input_tensor = output
        return output


class Network:
    def __init__(self):
        pass

    def get_network(self):
        X = tf.placeholder(tf.float32, shape=(None, IMG_SIZE, IMG_SIZE, 3), name='input')

        cnn_layers_0 = CNNMnistLayer([32, 32, 64], kernel_size=5, name='cnn_layer_0')
        # batch_norm_layer = tf.layers.BatchNormalization()
        cnn_layers_1 = CNNMnistLayer([64, 128], kernel_size=3, name='cnn_layer_1')

        dense = tf.layers.Dense(1024, activation=tf.nn.relu)
        logits = tf.layers.Dense(10)

        mnist_nn = cnn_layers_0(X)
        mnist_nn = cnn_layers_1(mnist_nn)
        mnist_nn = tf.reshape(mnist_nn, [-1, 8 * 8 * 128])

        mnist_nn = dense(mnist_nn)
        mnist_nn = logits(mnist_nn)
        return mnist_nn

