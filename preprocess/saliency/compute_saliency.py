import tensorflow as tf
import skimage
import numpy as np
import core
import cv2
import matplotlib.pyplot as plt

dataset_dir = './data/saliency_dataset/'
save_dir = '../data/saliency_dataset/'
model_checkpoint_path = '../model/pilotnetCloudV1'

if __name__ == '__main__':
    i = 0

    # construct model
    model = tf.keras.models.load_model(model_checkpoint_path)

    # function call to compute saliency map
    def call_model_function(images, call_model_args=None, expected_keys=None):
        images = tf.convert_to_tensor(images)
        with tf.GradientTape() as tape:
            if expected_keys == [core.base.INPUT_OUTPUT_GRADIENTS]:
                tape.watch(images)
                output_layer = model(images)
                gradients = np.array(tape.gradient(output_layer, images))
                return {core.base.INPUT_OUTPUT_GRADIENTS: gradients}
            else:
                conv_layer, output_layer = model(images)
                gradients = np.array(tape.gradient(output_layer, conv_layer))
                return {core.base.CONVOLUTION_LAYER_VALUES: conv_layer,
                        core.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}

    # Construct the saliency object. This alone doesn't do anthing.
    blur_ig = core.BlurIG()

    # compute the saliency maps for the first 2000 images of the dataset
    while i < 2000:
        full_image = skimage.io.imread(dataset_dir + str(i) + ".jpg", as_gray=False)
        sal_image = skimage.transform.resize(full_image, [66, 200])
        image = sal_image / 255.0
        image = np.expand_dims(image, axis=0)

        # get the values of the saliency map
        blur_ig_mask_3d = blur_ig.GetMask(sal_image, call_model_function, call_model_args=None, batch_size=1)
        # Call the visualization methods to convert the 3D tensors to 2D grayscale.
        blur_ig_mask_grayscale = core.VisualizeImageGrayscale(blur_ig_mask_3d)

        # transform the received values to show them as a heatmap "on top" of the real image
        foreground_image = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)
        background_image = skimage.transform.resize(blur_ig_mask_grayscale, [full_image.shape[0], full_image.shape[1]])
        background_image = (background_image * 255).astype(np.uint8)
        colormap = plt.get_cmap('jet')
        heatmap = (colormap(background_image) * 2 ** 8).astype(np.uint8)[:, :, :3]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        background = cv2.addWeighted(foreground_image, 1, heatmap, 0.4, 0)

        cv2.imwrite(save_dir + str(i) + ".jpg", background)

        i += 1
