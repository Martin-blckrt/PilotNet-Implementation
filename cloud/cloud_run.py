import tensorflow_cloud as tfc
import tensorflow as tf

tfc.run(
    requirements_txt="requirements.txt",
    entry_point='main.py',
    docker_image_bucket_name='pilotnet_bucket',
    chief_config=tfc.COMMON_MACHINE_CONFIGS['P100_4X'],
)
