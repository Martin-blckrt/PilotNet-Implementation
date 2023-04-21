import linecache
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils.image_dataset import load_image
import numpy as np
from preprocess.SteeringImageDB import SteeringImageDB
from model.PilotNet import PilotNet
import saliency.core as core
import skimage

dataset_dir = 'data/driving_dataset/train_dataset'
test_dataset_dir = './data/driving_dataset/mini_driving_dataset'
log_directory = './logs/'

name = "myPilotNet"

resize_width = 200
resize_height = 66

def call_model_function(images, call_model_args=None, expected_keys=None):
    images = tf.convert_to_tensor(images)
    with tf.GradientTape() as tape:
        if expected_keys==[core.base.INPUT_OUTPUT_GRADIENTS]:
            tape.watch(images)
            output_layer = model(images)
            gradients = np.array(tape.gradient(output_layer, images))
            return {core.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            conv_layer, output_layer = model(images)
            gradients = np.array(tape.gradient(output_layer, conv_layer))
            return {core.base.CONVOLUTION_LAYER_VALUES: conv_layer,
                    core.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}

i = 1000

# Load data.
og_img = skimage.io.imread('./data/train_dataset/train_dataset/' + str(i) + '.jpg')
img = skimage.transform.resize(og_img, [resize_height, resize_width])

line = linecache.getline(
    r'/data/driving_dataset/test_dataset/data.txt',
            i + 1)
degrees = float(line.split()[1].split(',')[0])

model = tf.keras.models.load_model(r"/model\saves\pilotnetCloudV1")

# Construct the saliency object. This alone doesn't do anthing.
xrai_object = core.XRAI()
# Compute XRAI attributions with default parameters
xrai_attributions = xrai_object.GetMask(img, call_model_function, call_model_args=None, batch_size=20)

# Construct the saliency object. This alone doesn't do anthing.
blur_ig = core.BlurIG()
# Compute the Blur IG mask and Smoothgrad+BlurIG mask.
blur_ig_mask_3d = blur_ig.GetMask(img, call_model_function, call_model_args=None, batch_size=20)
# Call the visualization methods to convert the 3D tensors to 2D grayscale.
blur_ig_mask_grayscale = core.VisualizeImageGrayscale(blur_ig_mask_3d)

# blur_ig_mask_grayscale = skimage.color.gray2rgb(skimage.transform.resize(blur_ig_mask_grayscale, [og_img.shape[0], og_img.shape[1]]))
blur_ig_mask_grayscale = skimage.transform.resize(blur_ig_mask_grayscale, [og_img.shape[0], og_img.shape[1]])

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(og_img)
j = axes[1].imshow(blur_ig_mask_grayscale, cmap="jet", alpha=0.8)
fig.colorbar(j)
axes[2].imshow(og_img)
axes[2].imshow(blur_ig_mask_grayscale, cmap="jet", alpha=0.4)

plt.show()

"""
img = tf.keras.preprocessing.image.img_to_array(img)
img = np.reshape(img, (1, 66, 200, 3))
images = tf.Variable(img, dtype=float)

with tf.GradientTape() as tape:
    pred = model(images, training=False)
    mse = tf.keras.losses.MeanSquaredError()
    print(degrees, pred)
    loss = mse(tf.convert_to_tensor(degrees), pred)

grads = tape.gradient(loss, images)
dgrad_abs = tf.math.abs(grads)
dgrad_max_ = np.max(dgrad_abs, axis=3)[0]

## normalize to range between 0 and 1
arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
grad_eval = skimage.transform.resize(grad_eval, [og_img.shape[0], og_img.shape[1]])

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(og_img)
j = axes[1].imshow(grad_eval, cmap="jet", alpha=0.8)
fig.colorbar(j)
axes[2].imshow(og_img)
axes[2].imshow(grad_eval, cmap="jet", alpha=0.2)

plt.show()
"""
