import numpy as np
import random
import skimage
from PIL import Image
from keras.preprocessing import image


class SteeringImageDB(object):
    """Preprocess images of the road ahead and steering angles."""

    def __init__(self, data_directory, width, height):
        imgs = []
        angles = []

        self.width = width
        self.height = height

        # read data.txt
        data_path = data_directory + "/"
        with open(data_path + "data.txt") as f:
            for line in f:
                imgs.append(data_path + line.split()[0])
                """ the paper by Nvidia uses the inverse of the turning radius, 
                but steering wheel angle is proportional to the inverse of turning radius
                so the steering wheel angle in radians is used as the output """
                angles.append(float(line.split()[1]) * np.pi / 180)

                # flip the image horizontally to augment dataset
                '''
                imgs.append(image.load_img(data_path + line.split()[0]).transpose(Image.FLIP_LEFT_RIGHT))
                angles.append(-(float(line.split()[1]) * np.pi / 180))                
                '''

        # shuffle list of images
        img_angle_pack = list(zip(imgs, angles))
        random.shuffle(img_angle_pack)
        imgs, angles = zip(*img_angle_pack)

        # get number of images
        self.num_images = len(imgs)

        # split 80-20 train/val
        self.train_imgs = imgs[:int(self.num_images * 0.8)]
        self.train_angles = angles[:int(self.num_images * 0.8)]

        self.val_imgs = imgs[-int(self.num_images * 0.2):]
        self.val_angles = angles[-int(self.num_images * 0.2):]

        self.num_train_images = len(self.train_imgs)
        self.num_val_images = len(self.val_imgs)

        print(
            "Data contains %d images : %d train, %d validation" % (
                self.num_images, self.num_train_images, self.num_val_images))

    def load_data(self, data_type):
        imgs = []
        angles = []

        if data_type == 'train':
            for i in range(0, self.num_train_images):
                imgs.append(
                    skimage.transform.resize(skimage.io.imread(self.train_imgs[i]), [self.height, self.width]) / 255.0)
                angles.append(self.train_angles[i])

        elif data_type == 'val':
            for i in range(0, self.num_val_images):
                imgs.append(
                    skimage.transform.resize(skimage.io.imread(self.val_imgs[i]), [self.height, self.width]) / 255.0)
                angles.append(self.val_angles[i])

        return np.array(imgs), np.array(angles)
