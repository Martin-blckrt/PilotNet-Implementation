import tensorflow as tf
import scipy.misc
from nets.pilotNet import PilotNet
import cv2
from subprocess import call

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'steer_image', './data/.logo/steering_wheel_image.jpg',
    """Steering wheel image to show corresponding steering wheel angle.""")

"""model from nvidia's training"""
tf.app.flags.DEFINE_string(
    'model_file', './data/models/nvidia/model.ckpt',
    """Path to the model parameter file.""")

# model from implemented training
# tf.app.flags.DEFINE_string(
#     'model', './data/model_save/model.ckpt',
#     """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'dataset_dir', './data/datasets/driving_dataset',
    """Directory that stores input recored front view images.""")

WIN_MARGIN_LEFT = 240
WIN_MARGIN_TOP = 240
WIN_MARGIN_BETWEEN = 180
WIN_WIDTH = 480

if __name__ == '__main__':
    img = cv2.imread(FLAGS.steer_image, 0)
    rows,cols = img.shape

    # Visualization init
    cv2.namedWindow("Steering Wheel", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Steering Wheel", WIN_MARGIN_LEFT, WIN_MARGIN_TOP)
    cv2.namedWindow("Scenario", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Scenario", WIN_MARGIN_LEFT+cols+WIN_MARGIN_BETWEEN, WIN_MARGIN_TOP)

    with tf.Graph().as_default():
        smoothed_angle = 0
        i = 0

        # construct model
        model = PilotNet()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # restore model variables
            saver.restore(sess, FLAGS.model_file)

            while(cv2.waitKey(10) != ord('q')):
                full_image = scipy.misc.imread(FLAGS.dataset_dir + "/" + str(i) + ".jpg", mode="RGB")
                image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0

                steering = sess.run(
                    model.steering,
                    feed_dict={
                        model.image_input: [image],
                        model.keep_prob: 1.0
                    }
                )

                degrees = steering[0][0] * 180.0 / scipy.pi
                call("clear")
                print("Predicted steering angle: " + str(degrees) + " degrees")
                # convert RGB due to dataset format
                cv2.imshow("Scenario", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
                print("Scenario image size: {} x {}").format(full_image.shape[0], full_image.shape[1])

                # make smooth angle transitions by turning the steering wheel based on the difference of the current angle
                # and the predicted angle
                smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
                M = cv2.getRotationMatrix2D((cols/2,rows/2), -smoothed_angle, 1)
                dst = cv2.warpAffine(img,M,(cols,rows))
                cv2.imshow("Steering Wheel", dst)

                i += 1

    cv2.destroyAllWindows()