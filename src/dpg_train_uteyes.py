#!/usr/bin/env python3
"""Main script for training the DPG model for within-MPIIGaze evaluations."""
"""v0.0[insfan][5/25/2019] ain script for training the DPG model for within UnityEyes evaluations"""
import argparse

import coloredlogs
import tensorflow as tf

if __name__ == '__main__':

    # Set global log level
    parser = argparse.ArgumentParser(description='Train the Deep Pictorial Gaze model.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )


    # Initialize Tensorflow session
    gpu_available = False
    try:
        gpus = [d for d in device_lib.list_local_devices(config=session_config)
                if d.device_type == 'GPU']
        gpu_available = len(gpus) > 0
    except:
        pass
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    gpu_options = tf.GPUOptions(allow_growth=True)  # Dynamic application for Graph memory
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:

        # Declare some parameters
        batch_size = 32
        person_id = 'p00'
        # Define training data source
        from datasources import UnityEyes_GazeMap
        from datasources import HDF5Source

        # Define model
        from models import DPG
        model = DPG(
            session,
            learning_schedule=[
                {
                    'loss_terms_to_optimize': {
                        'combined_loss': ['hourglass', 'densenet'],
                    },
                    'metrics': ['gaze_mse', 'gaze_ang'],
                    'learning_rate': 0.0002,
                },                      # Comma to convert to tuple type
            ],
            # extra_tags=[person_id],

            # Data sources for training (and testing).

            train_data={
                'uteyes': UnityEyes_GazeMap(
                    session,
                    batch_size=batch_size,
                    data_format='NCHW',
                    unityeyes_path='../datasets/UnityEyes//UnityEyes10000',
                    min_after_dequeue=1000,
                    # generate_heatmaps=True,
                    shuffle=True,
                    testing=False,
                    staging=True,
                    eye_image_shape=(90, 150),
                    # heatmaps_scale=1.0 / elg_first_layer_stride,
                ),
            },
            test_data={
                'mpi': HDF5Source(
                    session,
                    data_format='NCHW',
                    batch_size=batch_size,
                    keys_to_use=['test/' + person_id],
                    hdf_path='../datasets/MPIIGaze.h5',
                    eye_image_shape=(90, 150),
                    testing=True,
                ),
            },
        )

        # Train this model for a set number of epochs
        model.train(
            num_epochs=20,
        )

        model.__del__()
        session.close()
        del session
