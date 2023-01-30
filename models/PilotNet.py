import tensorflow as tf
import keras

num_epochs = 30
test_num_epochs = 5

batch_size = 128
steps_to_validate = 10
steps_per_epoch = 10


class PilotNet(keras.Model):
    def __init__(self, learning_rate, width, height, name=None):
        super().__init__(name=name)
        self.image_height = height
        self.image_width = width
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        inputs = keras.Input(name='input_shape', shape=(self.image_height, self.image_width, 3))

        # convolutional feature maps
        x = tf.keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(inputs)
        x = tf.keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)

        # flatten layer
        x = tf.keras.layers.Flatten()(x)

        # fully connected layers with dropouts for overfit protection
        x = tf.keras.layers.Dense(units=1152, activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=0.1)(x)
        x = tf.keras.layers.Dense(units=100, activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=0.1)(x)
        x = tf.keras.layers.Dense(units=50, activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=0.1)(x)
        x = tf.keras.layers.Dense(units=10, activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=0.1)(x)
        x = tf.keras.layers.Dense(units=1, activation='linear')(x)

        # derive steering angle value from single output layer by point multiplication
        steering_angle = tf.keras.layers.Lambda(lambda l: tf.multiply(tf.atan(l), 2), name='steering_angle')(x)

        # build and compile model
        model = keras.Model(inputs=[inputs], outputs=[steering_angle])
        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss={'steering_angle': 'mse'},
            metrics=['accuracy']
        )
        model.summary()
        return model

    def train(self, dataset, filename):
        size = int(dataset.num_train_images / 10)

        cb = tf.keras.callbacks.TensorBoard(log_dir="./logs")

        # fit data to model for training
        history = self.model.fit(x=dataset.load_train_batch(size)[0],
                                 y=dataset.load_train_batch(size)[1], batch_size=batch_size,
                                 epochs=test_num_epochs, verbose=1,
                                 validation_data=dataset.load_val_batch(size),
                                 callbacks=[cb])

        print(history)

        # test the model by fitting the test data
        stats = self.model.evaluate(x=dataset.load_test_batch(size)[0],
                                    y=dataset.load_test_batch(size)[1], verbose=1)
        # print the stats
        print(f'Model accuracy: {stats[1]}\nModel loss: {stats[0]}')
        input('\nPress [ENTER] to continue...')
        # save the trained model
        self.model.save(f"models/{filename}.h5")
