import numpy as np
import random
import skimage


class SteeringImageDB(object):
    """Preprocess images of the road ahead and steering angles."""

    def __init__(self, data_directory, width, height):
        imgs = []
        angles = []

        self.width = width
        self.height = height

        # points to the end of the last batch, train & val
        self.train_batch_pointer = 0
        self.val_batch_pointer = 0
        self.test_batch_pointer = 0

        # read data.txt
        data_path = data_directory + "/"
        with open(data_path + "data.txt") as f:
            for line in f:
                imgs.append(data_path + line.split()[0])
                """ the paper by Nvidia uses the inverse of the turning radius, 
                but steering wheel angle is proportional to the inverse of turning radius
                so the steering wheel angle in radians is used as the output """
                angles.append(float(line.split()[1]) * np.pi / 180)

        # shuffle list of images
        img_angle_pack = list(zip(imgs, angles))
        random.shuffle(img_angle_pack)
        imgs, angles = zip(*img_angle_pack)

        # get number of images
        self.num_images = len(imgs)

        # split 80-10-10 train/val
        self.train_imgs = imgs[:int(self.num_images * 0.8)]
        self.train_angles = angles[:int(self.num_images * 0.8)]

        self.rest_imgs = imgs[-int(self.num_images * 0.2):]
        self.rest_angles = angles[-int(self.num_images * 0.2):]
        self.num_rest = len(self.rest_imgs)

        self.val_imgs = imgs[:int(self.num_rest * 0.5)]
        self.val_angles = angles[:int(self.num_rest * 0.5)]

        self.test_imgs = imgs[-int(self.num_rest * 0.5):]
        self.test_angles = angles[-int(self.num_rest * 0.5):]

        self.num_train_images = len(self.train_imgs)
        self.num_val_images = len(self.val_imgs)
        self.num_test_images = len(self.test_imgs)

    def load_train_batch(self, batch_size):
        batch_imgs = []
        batch_angles = []
        for i in range(0, batch_size):
            batch_imgs.append(skimage.transform.resize(
                skimage.io.imread(self.train_imgs[(self.train_batch_pointer + i) % self.num_train_images]),
                [self.height, self.width]) / 255.0)
            batch_angles.append([self.train_angles[(self.train_batch_pointer + i) % self.num_train_images]])
        self.train_batch_pointer += batch_size
        return np.array(batch_imgs), np.array(batch_angles)

    def load_val_batch(self, batch_size):
        batch_imgs = []
        batch_angles = []
        for i in range(0, batch_size):
            batch_imgs.append(skimage.transform.resize(
                skimage.io.imread(self.val_imgs[(self.val_batch_pointer + i) % self.num_val_images]),
                [self.height, self.width]) / 255.0)
            batch_angles.append([self.val_angles[(self.val_batch_pointer + i) % self.num_val_images]])
        self.val_batch_pointer += batch_size
        return np.array(batch_imgs), np.array(batch_angles)

    def load_test_batch(self, batch_size):
        batch_imgs = []
        batch_angles = []
        for i in range(0, batch_size):
            batch_imgs.append(skimage.transform.resize(
                skimage.io.imread(self.test_imgs[(self.test_batch_pointer + i) % self.num_test_images]),
                [self.height, self.width]) / 255.0)
            batch_angles.append([self.test_angles[(self.test_batch_pointer + i) % self.num_test_images]])
        self.test_batch_pointer += batch_size
        return np.array(batch_imgs), np.array(batch_angles)
