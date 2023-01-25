import tensorflow as tf


class convLayer(tf.Module):
    def __init__(self, w_shape, b_shape, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(tf.random.truncated_normal(w_shape, stddev=0.1), name='w', trainable=True)
        self.b = tf.Variable(tf.constant(0.1, shape=b_shape), name='b', trainable=True)

    def __call__(self, x, stride):
        y = tf.nn.conv2d(x, self.w, strides=[1, stride, stride, 1], padding='SAME') + self.b
        return tf.nn.relu(y)


class FCL(tf.Module):
    def __init__(self, w_shape, b_shape, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(tf.random.truncated_normal(w_shape, stddev=0.1), name='w')
        self.b = tf.Variable(tf.constant(0.1, shape=b_shape), name='b')

    def __call__(self, x, keep_prob, last=False):
        if last:
            y = tf.atan(tf.matmul(x, self.w) + self.b)  # scale the atan output
            return tf.multiply(y, 2)
        else:
            y = tf.nn.relu(tf.matmul(x, self.w) + self.b)
            return tf.nn.dropout(y, rate=keep_prob)


class PilotNet(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.conv_1 = convLayer(w_shape=[5, 5, 3, 24], b_shape=[24], name="conv1")
        self.conv_2 = convLayer(w_shape=[5, 5, 24, 36], b_shape=[36], name="conv2")
        self.conv_3 = convLayer(w_shape=[5, 5, 36, 48], b_shape=[48], name="conv3")
        self.conv_4 = convLayer(w_shape=[3, 3, 48, 64], b_shape=[64], name="conv4")
        self.conv_5 = convLayer(w_shape=[3, 3, 64, 64], b_shape=[64], name="conv5")

        self.fcl_1 = FCL(w_shape=[1344, 1164], b_shape=[1164], name="fcl1")
        self.fcl_2 = FCL(w_shape=[1164, 100], b_shape=[100], name="fcl2")
        self.fcl_3 = FCL(w_shape=[100, 50], b_shape=[50], name="fcl3")
        self.fcl_4 = FCL(w_shape=[50, 10], b_shape=[10], name="fcl4")
        self.fcl_5 = FCL(w_shape=[10, 1], b_shape=[1], name="fcl5")

    def __call__(self, x, keep_prob=0.8):
        x = self.conv_1(x, 2)
        x = self.conv_2(x, 2)
        x = self.conv_3(x, 2)
        x = self.conv_4(x, 2)
        x = self.conv_5(x, 2)

        x = tf.reshape(x, [-1, 1344])

        x = self.fcl_1(x, keep_prob)
        x = self.fcl_2(x, keep_prob)
        x = self.fcl_3(x, keep_prob)
        x = self.fcl_4(x, keep_prob)
        x = self.fcl_5(x, keep_prob, True)

        return x
