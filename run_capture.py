import skimage
import tensorflow as tf
import sys
from models.PilotNet import PilotNet
import cv2
from subprocess import call
from absl import flags
import numpy as np

FLAGS = flags.FLAGS
FLAGS(sys.argv)

"""model from nvidia's training"""
flags.DEFINE_string(
    'model', './data/models/nvidia/model.ckpt',
    """Path to the model parameter file.""")
# generated model after training
# flags.DEFINE_string(
#     'model', './data/models/model.ckpt',
#     """Path to the model parameter file.""")

flags.DEFINE_string(
    'steer_image', './data/.logo/steering_wheel_image.jpg',
    """Steering wheel image to show corresponding steering wheel angle.""")

WIN_MARGIN_LEFT = 240
WIN_MARGIN_TOP = 240
WIN_MARGIN_BETWEEN = 180
WIN_WIDTH = 480

if __name__ == '__main__':
    img = cv2.imread(FLAGS.steer_image, 0)
    rows, cols = img.shape

    cap = cv2.VideoCapture(0)

    # Visualization init
    cv2.namedWindow("Steering Wheel", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Steering Wheel", WIN_MARGIN_LEFT, WIN_MARGIN_TOP)
    cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Capture", WIN_MARGIN_LEFT + cols + WIN_MARGIN_BETWEEN, WIN_MARGIN_TOP)

    with tf.Graph().as_default():
        smoothed_angle = 0
        i = 0

        # construct model
        model = PilotNet(name="myPilotNet")

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # restore model variables
            saver.restore(sess, FLAGS.model)

            while (cv2.waitKey(10) != ord('q')):
                ret, frame = cap.read()
                image = skimage.transform.resize(frame, [66, 200]) / 255.0

                steering = sess.run(
                    model.steering,
                    feed_dict={
                        model.image_input: [image],
                        model.keep_prob: 1.0
                    }
                )

                degrees = steering[0][0] * 180.0 / np.pi
                call("clear")
                print("Predicted steering angle: " + str(degrees) + " degrees")

                cv2.imshow("Capture", frame)
                print("Captured image size: {} x {}").format(frame.shape[0], frame.shape[1])

                # make smooth angle transitions by turning the steering wheel based on the difference of the current angle
                # and the predicted angle
                smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (
                            degrees - smoothed_angle) / abs(degrees - smoothed_angle)
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
                dst = cv2.warpAffine(img, M, (cols, rows))
                cv2.imshow("Steering Wheel", dst)

                i += 1

    cap.release()
    cv2.destroyAllWindows()
