import tensorflow as tf
from preprocess.SteeringImageDB import SteeringImageDB
from models.PilotNet import PilotNet

dataset_dir = './data/driving_dataset/driving_dataset'
test_dataset_dir = './data/driving_dataset/mini_driving_dataset'
log_directory = './logs/'

name = "myPilotNet"

learning_rate = 1e-4

resize_width = 200
resize_height = 66

dataset = SteeringImageDB(test_dataset_dir, resize_width, resize_height)

pilotNetModel = PilotNet(learning_rate=learning_rate, input_shape=(66, 200, 3), name=name)

pilotNetModel.train(dataset=dataset, filename='./models/' + name)
