from preprocess.SteeringImageDB import SteeringImageDB
from model.PilotNet import PilotNet

dataset_dir = './data/train_dataset/train_dataset'
log_directory = './logs/'

# define model name
name = "myPilotNet"
# define Google Cloud Platform bucket
GCP_BUCKET = 'pilotnet_bucket'

learning_rate = 1e-4

resize_width = 200
resize_height = 66

# if training is local, load the dataset
dataset = SteeringImageDB(dataset_dir, resize_width, resize_height)

# define model
pilotNetModel = PilotNet(learning_rate=learning_rate, input_shape=(66, 200, 3), name=name)

# train the model
pilotNetModel.train(dataset=dataset, filename='./model/' + name)
