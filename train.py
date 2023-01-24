import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
from preprocess.SteeringImageDB import SteeringImageDB
from models.PilotNet import PilotNet

FLAGS = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_string(
    'dataset_dir', './data/driving_dataset/driving_dataset',
    """Directory that stores input recored front view images and steering wheel angles.""")
tf.compat.v1.flags.DEFINE_bool(
    'clear_log', False,
    """force to clear old logs if exist.""")
tf.compat.v1.flags.DEFINE_string(
    'log_directory', './logs/',
    """Directory for training logs, including training summaries as well as training model checkpoint.""")
tf.compat.v1.flags.DEFINE_float(
    'L2NormConst', 1e-3,
    """L2-Norm const value for loss computation.""")
tf.compat.v1.flags.DEFINE_float(
    'learning_rate', 1e-4,
    """Learning rate determines the incremental steps of the values to find the best weight values.""")
tf.compat.v1.flags.DEFINE_integer(
    'num_epochs', 30,
    """The numbers of epochs for training, train over the dataset about 30 times.""")
tf.compat.v1.flags.DEFINE_integer(
    'batch_size', 128,
    """The numbers of training examples present in a single batch for every training.""")


def train(argv=None):
    """Train PilotNet model"""

    # delete old logs
    if FLAGS.clear_log:
        if tf.gfile.Exists(FLAGS.log_directory):
            tf.gfile.DeleteRecursively(FLAGS.log_directory)
        tf.gfile.MakeDirs(FLAGS.log_directory)

    with tf.Graph().as_default():
        # construct model
        model = PilotNet()

        # images of the road ahead and steering angles in random order
        dataset = SteeringImageDB(FLAGS.dataset_dir)

        '''
        Training with stochastic gradient descent (Adaptive Moment Estimation, Adam) : find the best weight values and
        biases to minimize the output error
          1. compute_gradients(loss, <list of variables>)
          2. apply_gradients(<list of variables>)
        '''

        train_vars = tf.compat.v1.trainable_variables()

        # define loss
        def loss():
            return tf.math.reduce_mean(tf.math.square(tf.math.subtract(model.y_, model.steering))) \
                + tf.math.add_n([tf.nn.l2_loss(v) for v in train_vars]) * FLAGS.L2NormConst

        cost = tf.math.reduce_mean(tf.math.square(tf.math.subtract(model.y_, model.steering))) + tf.math.add_n(
            [tf.nn.l2_loss(v) for v in train_vars]) * FLAGS.L2NormConst

        optimizer = tf.optimizers.Adam(FLAGS.learning_rate).minimize(loss, var_list=train_vars)

        '''
        TensorFlow's V1 checkpoint format has been deprecated.
        Consider switching to the more efficient V2 format, now on by default.
        '''
        # saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)
        saver = tf.train.Checkpoint()

        # op to write logs to Tensorboard
        summary_writer = tf.summary.create_file_writer(FLAGS.log_directory)
        save_model_path = FLAGS.log_directory + "/checkpoint/"

        print("Run the command line:\n" \
              "--> tensorboard --logdir={} " \
              "\nThen open http://0.0.0.0:6006/ into your web browser".format(FLAGS.log_directory))

        for epoch in range(FLAGS.num_epochs):
            # Throughout the neural network run, you pass two things: the loss calculation and the optimization step

            # the number of batches is equal to number of iterations for one epoch.
            num_batches = int(dataset.num_images / FLAGS.batch_size)
            for batch in range(num_batches):
                imgs, angles = dataset.load_train_batch(FLAGS.batch_size)

                # Run optimization op (backprop) and cost op (to get loss value)
                loss()
                optimizer
                feed_dict = {
                    model.image_input: imgs,
                    model.y_: angles,
                    model.keep_prob: 0.8
                }

                if batch % 10 == 0:
                    imgs, angles = dataset.load_val_batch(FLAGS.batch_size)
                    loss_value = loss()
                    feed_dict = {
                        model.image_input: imgs,
                        model.y_: angles,
                        model.keep_prob: 1.0
                    }
                    # A training step is one gradient update, in one step batch_size many examples are processed.
                    print("Epoch: %d, Step: %d, Loss: " % (epoch, epoch * FLAGS.batch_size + batch),
                          tf.reduce_mean(tf.square(tf.subtract(model.y_, model.steering))) \
                          + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * FLAGS.L2NormConst)

                # write logs at every iteration
                with summary_writer.as_default(step=batch):
                    # create a summary to monitor cost tensor
                    tf.summary.scalar("loss", loss())

                # Save the model checkpoint periodically.
                if batch % FLAGS.batch_size == 0:
                    if not os.path.exists(save_model_path):
                        os.makedirs(save_model_path)
                    checkpoint_path = os.path.join(save_model_path, "model.ckpt")
                    filename = saver.save(checkpoint_path)

            print("Model saved in file: %s" % filename)


if __name__ == '__main__':
    train()
