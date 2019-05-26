#!/usr/bin/env python3
"""Main script for training the DPG model for within-MPIIGaze evaluations."""
"""v0.0[insfan][5/25/2019] Main script for training the Densenet model for within-MPIIGaze npz file evaluations"""
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

    for i in range(0, 1):
        # Specify which people to train on, and which to test on
        person_id = 'p%02d' % i
        other_person_ids = ['p%02d' % j for j in range(15) if i != j]

        # Initialize Tensorflow session
        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.INFO)
        gpu_options = tf.GPUOptions(allow_growth=True)  # Dynamic application for Graph memory
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:

            # Declare some parameters
            batch_size = 4

            # Define training data source
            from datasources import NPZSource

            # Define model
            from models import Densenet
            model = Densenet(
                session,
                learning_schedule=[
                    {
                        'loss_terms_to_optimize': {
                            'combined_loss': ['densenet'],
                        },
                        'metrics': ['gaze_mse'],
                        'learning_rate': 0.001,
                    },                      # Comma to convert to tuple type
                ],
                extra_tags=[person_id],

                # Data sources for training (and testing).
                train_data={
                    'mpi': NPZSource(
                        session,
                        data_format='NCHW' if gpu_available else 'NHWC',
                        batch_size=batch_size,
                        # keys_to_use=['train/' + s for s in other_person_ids],
                        npz_path='../datasets/MPIIFaceGaze_eyes_p00_100.npz',
                        eye_image_shape=(120, 120),
                        testing=False,
                        min_after_dequeue=50,
                        staging=True,
                        shuffle=True,
                    ),
                },
                # test_data={
                #     'mpi': HDF5IrisSource(
                #         session,
                #         data_format='NCHW' if gpu_available else 'NHWC',
                #         batch_size=batch_size,
                #         keys_to_use=['test/' + person_id],
                #         hdf_path='../datasets/MPIIGaze.h5',
                #         eye_image_shape=(90, 150),
                #         testing=True,
                #     ),
                # },
            )

            # Train this model for a set number of epochs
            model.train(
                num_epochs=100,
            )

            model.__del__()
            session.close()
            del session
