import linecache
import tensorflow as tf
import skimage
import scipy.misc
import numpy as np
from tensorflow import keras
import cv2

MODEL = '/models/nvidia/model.ckpt'
STEER_IMAGE = r'C:\Users\marti\PycharmProjects\PilotNet-Implementation\data\.logo\steering-wheel.png'
dataset_dir = './data/driving_dataset/driving_dataset'
other_dataset_dir = './data/driving_dataset/other_driving_dataset'

model_checkpoint_path = r"C:\Users\marti\PycharmProjects\PilotNet-Implementation\models\saves\pilotnetCloudV1"

WIN_MARGIN_LEFT = 130
WIN_MARGIN_TOP = 240
WIN_MARGIN_BETWEEN = 180
WIN_WIDTH = 480

if __name__ == '__main__':
    img = cv2.imread(STEER_IMAGE, cv2.IMREAD_UNCHANGED)
    rows, cols, _ = img.shape

    # Visualization initd
    cv2.namedWindow("Scenario", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Scenario", 1200, 700)
    cv2.namedWindow("Wheel", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Wheel', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow("Wheel", 200, 200)
    cv2.moveWindow("Wheel", 670, 590)

    smoothed_angle = 0
    i = 0

    # construct model
    # model = keras.models.load_model(model_checkpoint_path)
    model = tf.keras.models.load_model(model_checkpoint_path)

    while True:
        full_image = skimage.io.imread(other_dataset_dir + "/" + str(i) + ".jpg", as_gray=False)
        image = skimage.transform.resize(full_image, [66, 200]) / 255.0
        image = np.expand_dims(image, axis=0)

        # Make a prediction
        prediction = model.predict(image)

        # steering = tf.math.argmax(prediction[0][0], axis=1, output_type=tf.int64)

        degrees = prediction[0][0] * 180.0 / scipy.pi
        """

        line = linecache.getline(
            r'C://Users/marti/PycharmProjects/PilotNet-Implementation/data/driving_dataset/driving_dataset/data.txt',
            i + 1)
        degrees = float(line.split()[1])
        if degrees == 0.0:
            degrees += 0.0000001
        """

        print("Predicted steering angle: " + str(round(degrees, 2)) + " degrees")
        # convert RGB due to dataset format
        cv2.imshow("Scenario", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))

        # make smooth angle transitions by turning the steering wheel based on the difference of the current angle
        # and the predicted angle
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
            degrees - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))

        cv2.imshow("Wheel", dst)

        i += 1

        if cv2.waitKey(15) != -1:
            break

cv2.destroyAllWindows()
