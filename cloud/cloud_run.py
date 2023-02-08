import tensorflow_cloud as tfc

tfc.run(
    requirements_txt="requirements.txt",
    entry_point='main.py',
    docker_image_bucket_name='pilotnet_bucket',
)