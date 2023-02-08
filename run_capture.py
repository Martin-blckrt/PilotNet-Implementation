import skimage
import tensorflow as tf
import sys
from models.PilotNet import PilotNet
import cv2
from subprocess import call
from absl import flags
import numpy as np

MODEL = '/models/nvidia/model.ckpt'
STEER_IMAGE = r'C:\Users\marti\PycharmProjects\PilotNet-Implementation\data\.logo\steering_wheel_image.png'

WIN_MARGIN_LEFT = 240
WIN_MARGIN_TOP = 240
WIN_MARGIN_BETWEEN = 180
WIN_WIDTH = 480

if __name__ == '__main__':
    img = cv2.imread(STEER_IMAGE)
    rows, cols, _ = img.shape

    cap = cv2.VideoCapture(0)

    # Visualization init
    cv2.namedWindow("Steering Wheel", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Steering Wheel", WIN_MARGIN_LEFT, WIN_MARGIN_TOP)
    cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Capture", WIN_MARGIN_LEFT + cols + WIN_MARGIN_BETWEEN, WIN_MARGIN_TOP)

    smoothed_angle = 0
    i = 0

    # construct model
    model = PilotNet(name="myPilotNet", learning_rate=1e-4, input_shape=(66, 200, 3))

    model

    while cv2.waitKey(0) != ord('q'):
        ret, frame = cap.read()
        image = skimage.transform.resize(frame, [66, 200]) / 255.0
"""
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
"""
cap.release()
cv2.destroyAllWindows()
