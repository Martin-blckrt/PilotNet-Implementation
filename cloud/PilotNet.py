from abc import ABC
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import datetime
import os
import tensorflow_cloud as tfc

batch_size = 128

GCP_BUCKET = 'pilotnet_bucket'
MODEL_PATH = "pilotnetCloudV1"


class PilotNet(keras.Model, ABC):
    def __init__(self, learning_rate, input_shape, name=None):
        super().__init__(name=name)
        self.i_shape = input_shape
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        inputs = keras.Input(name='input_shape', shape=self.i_shape)

        # convolutional feature maps
        x = tf.keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(inputs)
        x = tf.keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)

        # flatten layer
        x = tf.keras.layers.Flatten()(x)

        # fully connected layers with dropouts for overfit protection
        x = tf.keras.layers.Dense(units=100, activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=0.1)(x)
        x = tf.keras.layers.Dense(units=50, activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=0.1)(x)
        x = tf.keras.layers.Dense(units=10, activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=0.1)(x)
        steering_angle = tf.keras.layers.Dense(units=1, activation='linear')(x)

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
        checkpoint_path = os.path.join("gs://", GCP_BUCKET, MODEL_PATH, "save_at_{epoch}")
        tensorboard_path = os.path.join(
            "gs://", GCP_BUCKET, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        callbacks_list = [
            ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=False),
            EarlyStopping(monitor='val_accuracy', patience=5, verbose=0),
            TensorBoard(log_dir=tensorboard_path, histogram_freq=0, write_graph=False, write_images=False)
        ]

        x, y = dataset.load_data(data_type='train')

        if tfc.remote():
            epochs = 50
        else:
            epochs = 2
            callbacks_list = None

        # fit data to model for training
        history = self.model.fit(x=x,
                                 y=y,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=1,
                                 validation_data=dataset.load_data(data_type='val'),
                                 callbacks=callbacks_list)

        # save the trained model
        if tfc.remote():
            SAVE_PATH = os.path.join("gs://", GCP_BUCKET, MODEL_PATH)
            self.model.save(SAVE_PATH)
        else:
            self.model.save(f"models/{filename}.h5")
