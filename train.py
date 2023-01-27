import os
import tensorflow as tf
from preprocess.SteeringImageDB import SteeringImageDB
from models.PilotNet import PilotNet

# Define flags
dataset_dir = './data/driving_dataset/driving_dataset'
test_dataset_dir = './datasets/driving_dataset/mini_driving_dataset'
clear_log = False
log_dir = './logs/'
L2NormConst = 1e-3
learning_rate = 1e-4
num_epochs = 30
test_num_epochs = 5
batch_size = 128

if clear_log:
    if tf.io.gfile.exists(log_dir):
        tf.io.gfile.rmtree(log_dir)
    tf.io.gfile.mkdir(log_dir)

# construct model
model = PilotNet(name="myPilotNet")

# images of the road ahead and steering angles in random order
dataset = SteeringImageDB(dataset_dir)
print(model.trainable_variables)
# Define loss and optimizer
loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.steering))) + tf.add_n(
    [tf.nn.l2_loss(v) for v in model.trainable_variables]) * L2NormConst
optimizer = tf.optimizers.Adam(learning_rate)

# Create summary for monitoring loss
tf.summary.scalar("loss", loss)
summary_writer = tf.summary.create_file_writer(log_dir)

print("Starting...")

# Training loop
for epoch in range(num_epochs):
    num_batches = int(dataset.num_images / batch_size)
    for batch in range(num_batches):
        imgs, angles = dataset.load_train_batch(batch_size)
        with tf.GradientTape() as tape:
            predictions = model(imgs)
            current_loss = loss(angles, predictions)
        grads = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if batch % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar("loss", current_loss, step=batch)

    print(f"Epoch {epoch:2d}:")
    print("    ", current_loss)
