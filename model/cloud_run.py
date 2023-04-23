import tensorflow_cloud as tfc

# entry point to execute training with tensorflow cloud
tfc.run(
    entry_point='train.py',
    docker_image_bucket_name='pilotnet_bucket',
)
