import tensorflow as tf
import scipy.misc
from nets import model_nvidia
import cv2
import matplotlib.pyplot as plt
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_dir', './data/datasets/driving_dataset',
    """Directory that stores input recored front view images and steering wheel angles.""")

"""model from nvidia's training"""
tf.app.flags.DEFINE_string(
    'model_file', './data/models/nvidia/model.ckpt',
    """Path to the model parameter file.""")

def _generate_feature_image(feature_map, shape):
    dim = feature_map.shape[2]
    row_step = feature_map.shape[0]
    col_step = feature_map.shape[1]

    feature_image = np.zeros([row_step*shape[0], col_step*shape[1]])
    min = np.min(feature_map)
    max = np.max(feature_map)
    minmax = np.fabs(min - max)
    cnt = 0
    for row in range(shape[0]):
        row_idx = row_step * row
        row_idx_nxt = row_step * (row + 1)
        for col in range(shape[1]):
            col_idx = col_step * col
            col_idx_nxt = col_step * (col + 1)
            feature_image[row_idx:row_idx_nxt, col_idx:col_idx_nxt] = (feature_map[:, :, cnt] - min) * 1.0/minmax
            cnt += 1
    return feature_image

def show_activation(argv=None):
    """show the activations of the first two feature map layers"""

    # randomly choose an img from dataset
    full_image = scipy.misc.imread(FLAGS.dataset_dir + "/29649" + ".jpg", mode="RGB")
    # input planes: 3@66x200 & Normalization
    image = scipy.misc.imresize(full_image, [66, 200]) / 255.0

    fig = plt.figure('Visualization of Internal CNN State')
    plt.subplot(211)
    plt.title('Normalized input planes 3@66x200 to the CNN')
    plt.imshow(image)

    saver = tf.train.Saver()

    # model has been constructed from import
    # with tf.Graph().as_default():

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, FLAGS.model_file)
        print("Load session successfully")

        conv1act, conv2act, conv3act, conv4act, conv5act = sess.run(
            [model_nvidia.h_conv1, model_nvidia.h_conv2, model_nvidia.h_conv3, model_nvidia.h_conv4, model_nvidia.h_conv5],
            feed_dict={
                model_nvidia.x: [image]
            }
        )

        conv1img = _generate_feature_image(conv1act[0], [6, int(conv1act.shape[3]/6)])
        conv2img = _generate_feature_image(conv2act[0], [6, int(conv1act.shape[3]/6)])

        # get the mean, and supress the first(batch) dimension
        averageC5 = np.mean(conv5act, axis=3).squeeze(axis=0)
        averageC4 = np.mean(conv4act, axis=3).squeeze(axis=0)
        averageC3 = np.mean(conv3act, axis=3).squeeze(axis=0)
        averageC2 = np.mean(conv2act, axis=3).squeeze(axis=0)
        averageC1 = np.mean(conv1act, axis=3).squeeze(axis=0)

        # upscale
        averageC5up = scipy.misc.imresize(averageC5, [averageC4.shape[0], averageC4.shape[1]])
        multC45 = np.multiply(averageC5up, averageC4)
        multC45up = scipy.misc.imresize(multC45, [averageC3.shape[0], averageC3.shape[1]])
        multC34 = np.multiply(multC45up, averageC3)
        multC34up = scipy.misc.imresize(multC34, [averageC2.shape[0], averageC2.shape[1]])
        multC23 = np.multiply(multC34up, averageC2)
        multC23up = scipy.misc.imresize(multC23, [averageC1.shape[0], averageC1.shape[1]])
        multC12 = np.multiply(multC23up, averageC1)
        multC12up = scipy.misc.imresize(multC12, [image.shape[0], image.shape[1]])

        # normalize to [0,1], however, it did not show the salient map, the multC12up shows something like salient
        salient_mask = (multC12up - np.min(multC12up))/(np.max(multC12up) - np.min(multC12up))
        plt.subplot(223)
        plt.title('Activation of the \nfirst layer feature maps')
        plt.imshow(conv1img, cmap='gray')
        # plt.imshow(multC12up)

        plt.subplot(224)
        plt.title('Activation of the \nsecond layer feature maps')
        # plt.imshow(salient_mask)
        plt.imshow(conv2img, cmap='gray')

    plt.show()

if __name__ == '__main__':
    # run the train function
    tf.app.run(main=show_activation, argv=[])