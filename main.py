import tensorflow as tf
from preprocess.SteeringImageDB import SteeringImageDB
from models.PilotNet import PilotNet
import skimage

dataset_dir = './data/driving_dataset/driving_dataset'
test_dataset_dir = './data/driving_dataset/mini_driving_dataset'
log_directory = './logs/'

filename = './models/model_v'

learning_rate = 1e-4

input_width = 200
input_height = 66

dataset = SteeringImageDB(test_dataset_dir, input_width, input_height)

data_path = test_dataset_dir + "/"
with open(data_path + "data.txt") as f:
    for line in f:
        random_img = data_path + line.split()[0]

t = skimage.io.imread(random_img)
height = t.shape[0]
width = t.shape[1]
pilotNetModel = PilotNet(learning_rate=learning_rate, width=input_width, height=input_height, name="myPilotNet")

pilotNetModel.train(dataset=dataset, filename=filename)
