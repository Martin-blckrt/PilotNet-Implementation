from SteeringImageDB import SteeringImageDB
from PilotNet import PilotNet
import tensorflow_cloud as tfc

dataset_dir = './driving_dataset'
test_dataset_dir = '../data/driving_dataset/mini_driving_dataset'

name = "myPilotNet"
GCP_BUCKET = 'pilotnet_bucket'

learning_rate = 1e-4
resize_width = 200
resize_height = 66

if tfc.remote():
    dataset = SteeringImageDB(dataset_dir, resize_width, resize_height)
else:
    dataset = SteeringImageDB(test_dataset_dir, resize_width, resize_height)

pilotNetModel = PilotNet(learning_rate=learning_rate, input_shape=(resize_height, resize_width, 3), name=name)

pilotNetModel.train(dataset=dataset, filename='./models/' + name)
