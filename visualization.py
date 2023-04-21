import linecache
import tensorflow as tf
import skimage
import numpy as np
import cv2

steer_image_dir = './data/.logo/steering-wheel.png'
dataset_dir = './data/saliency_dataset/'
saliency_dataset_dir = './data/saliency_dataset/'
model_checkpoint_path = './model/pilotnetCloudV1'

WIN_MARGIN_LEFT = 130
WIN_MARGIN_TOP = 240
WIN_MARGIN_BETWEEN = 180
WIN_WIDTH = 480


WITH_MODEL = True
WITH_SALIENCY = True

if __name__ == '__main__':

    print("Press any key to stop the simulation")

    # load steering wheel image
    steer_img = cv2.imread(steer_image_dir, cv2.IMREAD_UNCHANGED)
    rows, cols, _ = steer_img.shape

    # Visualization init
    cv2.namedWindow("Scenario", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Scenario", 1200, 700)
    cv2.namedWindow("Wheel", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Wheel', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow("Wheel", 200, 200)
    cv2.moveWindow("Wheel", 670, 590)

    smoothed_angle = 0
    i = 0

    # construct model
    model = tf.keras.models.load_model(model_checkpoint_path)

    while i < 1000:
        # load the current image
        scenario_image = skimage.io.imread(dataset_dir + str(i) + ".jpg", as_gray=False)

        # pretreat it to feed it to the model
        prediction_image = np.expand_dims(skimage.transform.resize(scenario_image, [66, 200]) / 255.0, axis=0)

        # run the "perfect" angles or the models prediction
        if WITH_MODEL:
            # make the model predict an angle (and modify it to transform it to degrees)
            prediction = model(prediction_image)
            degrees = prediction.numpy()[0][0] * 180.0 / np.pi
        else:
            # get the perfect angle from the test dataset
            line = linecache.getline(dataset_dir + 'data.txt', i+1)
            degrees = float(line.split()[1].split(',')[0])
            # we have to divide by degrees later so degrees can't be 0
            if degrees == 0.0:
                degrees += 0.000000001

        # run with or without saliency map
        if WITH_SALIENCY:
            scenario_image = skimage.io.imread(saliency_dataset_dir + str(i) + ".jpg", as_gray=False)

        cv2.imshow("Scenario", cv2.cvtColor(scenario_image, cv2.COLOR_RGB2BGR))

        # smooth angle transitions by turning the wheel based on the difference of the current angle and the predicted angle
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
            degrees - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
        angled_wheel = cv2.warpAffine(steer_img, M, (cols, rows))

        cv2.imshow("Wheel", angled_wheel)

        i += 1

        if cv2.waitKey(1) != -1:
            break

    cv2.destroyAllWindows()
