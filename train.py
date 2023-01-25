import sys

import tensorflow as tf
import os
import numpy as np

from absl import flags
from preprocess.SteeringImageDB import SteeringImageDB
from models.PilotNet import PilotNet

FLAGS = flags.FLAGS
FLAGS(sys.argv)

flags.DEFINE_string(
    'dataset_dir', './data/driving_dataset/driving_dataset',
    """Directory that stores input recored front view images and steering wheel angles.""")
flags.DEFINE_bool(
    'clear_log', False,
    """force to clear old logs if exist.""")
flags.DEFINE_string(
    'log_directory', './logs/',
    """Directory for training logs, including training summaries as well as training model checkpoint.""")
flags.DEFINE_float(
    'L2NormConst', 1e-3,
    """L2-Norm const value for loss computation.""")
flags.DEFINE_float(
    'learning_rate', 1e-4,
    """Learning rate determines the incremental steps of the values to find the best weight values.""")
flags.DEFINE_integer(
    'num_epochs', 30,
    """The numbers of epochs for training, train over the dataset about 30 times.""")
flags.DEFINE_integer(
    'batch_size', 128,
    """The numbers of training examples present in a single batch for every training.""")


def training_loop(model):
    # images of the road ahead and steering angles in random order
    dataset = SteeringImageDB(FLAGS.dataset_dir)

    """Train PilotNet model"""
    # delete old logs
    if FLAGS.clear_log:
        if tf.gfile.Exists(FLAGS.log_directory):
            tf.gfile.DeleteRecursively(FLAGS.log_directory)
        tf.gfile.MakeDirs(FLAGS.log_directory)

    saver = tf.train.Checkpoint(model=model)
    save_model_path = FLAGS.log_directory + "/checkpoint/"

    # op to write logs to Tensorboard
    summary_writer = tf.summary.create_file_writer(FLAGS.log_directory)

    optimizer = tf.optimizers.Adam(FLAGS.learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

    for epoch in range(FLAGS.num_epochs):

        # the number of batches is equal to number of iterations for one epoch.
        num_batches = int(dataset.num_images / FLAGS.batch_size)
        for batch in range(num_batches):
            imgs, angles = dataset.load_train_batch(FLAGS.batch_size)

            with tf.GradientTape() as tape:
                logits = model(imgs)
                loss_value = loss_fn(np.array(angles), logits)

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(f"Epoch {epoch:2d}:")
        print("    ", loss_value)

        # write logs at every iteration
        with summary_writer.as_default(step=batch):
            # create a summary to monitor cost tensor
            tf.summary.scalar("loss", loss_value)

        # Save the model checkpoint periodically.
        if batch % FLAGS.batch_size == 0:
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            checkpoint_path = os.path.join(save_model_path, "model.ckpt")
            saver.write(checkpoint_path)

        print("Model saved in file: %s" % checkpoint_path)


print(f"Starting:")

# construct model
pilotNetModel = PilotNet(name="myPilotNet")

# Collect the history of W-values and b-values to plot later
weights = []
biases = []

training_loop(pilotNetModel)
