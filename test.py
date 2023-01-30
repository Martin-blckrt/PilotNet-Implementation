import tensorflow as tf
from preprocess.SteeringImageDB import SteeringImageDB
from models.PilotNet import PilotNet

test_dataset_dir = './data/driving_dataset/mini_driving_dataset'
batch_size = 128
model_save_path = './models/first_saved_model'
model_checkpoint_path = './models/first_checkpoint_model/checkpoint'

test_accuracy = tf.keras.metrics.Accuracy()

dataset = SteeringImageDB(test_dataset_dir)

pilotNetModel = PilotNet(name="myPilotNet")
checkpoint = tf.train.Checkpoint(model=pilotNetModel)
t = checkpoint.read(tf.train.latest_checkpoint(model_checkpoint_path))

num_batches = int(dataset.num_images / batch_size)
for batch in range(num_batches):
    imgs, angles = dataset.load_val_batch(batch_size)

    logits = checkpoint.model(imgs, 0.1)

    prediction = tf.math.argmax(logits, axis=1, output_type=tf.int64)
    test_accuracy(prediction, angles)

    print(prediction[0], angles[0])

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
