from abc import ABC

import tensorflow as tf


class PilotNet(tf.keras.Model, ABC):
    def __init__(self, name=None):
        super(PilotNet, self).__init__(name=name)

        self.w_fc1 = tf.Variable(tf.random.truncated_normal(shape=[5, 5, 3, 24], stddev=0.1))
        self.w_fc2 = tf.Variable(tf.random.truncated_normal(shape=[5, 5, 24, 36], stddev=0.1))
        self.w_fc3 = tf.Variable(tf.random.truncated_normal(shape=[5, 5, 36, 48], stddev=0.1))
        self.w_fc4 = tf.Variable(tf.random.truncated_normal(shape=[5, 5, 48, 64], stddev=0.1))
        self.w_fc5 = tf.Variable(tf.random.truncated_normal(shape=[5, 5, 64, 64], stddev=0.1))

        self.conv1 = tf.keras.layers.Conv2D(24, [5, 5, 3, 24], strides=(2, 2), padding='same', activation='relu',
                                            use_bias=True, bias_initializer=tf.constant(0.1, shape=[24]),
                                            weights=self.w_fc1)
        self.conv2 = tf.keras.layers.Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation='relu',
                                            use_bias=True, bias_initializer=tf.constant(0.1, shape=[36]),
                                            weights=self.w_fc2)
        self.conv3 = tf.keras.layers.Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation='relu',
                                            use_bias=True, bias_initializer=tf.constant(0.1, shape=[48]),
                                            weights=self.w_fc3)
        self.conv4 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu',
                                            use_bias=True, bias_initializer=tf.constant(0.1, shape=[64]),
                                            weights=self.w_fc4)
        self.conv5 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu',
                                            use_bias=True, bias_initializer=tf.constant(0.1, shape=[64]),
                                            weights=self.w_fc5)
        self.fc1 = tf.keras.layers.Dense(1164, activation='relu')
        self.fc2 = tf.keras.layers.Dense(100, activation='relu')
        self.fc3 = tf.keras.layers.Dense(50, activation='relu')
        self.fc4 = tf.keras.layers.Dense(10, activation='relu')
        self.fc5 = tf.keras.layers.Dense(1)
        self.dropout = tf.keras.layers.Dropout(0.8)

        self.x = tf.Variable(tf.zeros((66, 200, 3)), trainable=False, dtype=tf.float32)
        self.y_ = tf.Variable(tf.zeros((1, 1)), trainable=False, dtype=tf.float32)
        self.steering = tf.Variable(tf.zeros((1, 1)), trainable=False, dtype=tf.float32)

    def __call__(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = tf.reshape(x, [-1, 1344])

        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        x = self.fc3(x)
        x = self.dropout(x, training=training)
        x = self.fc4(x)
        x = self.dropout(x, training=training)
        x = self.fc5(x)

        return x
