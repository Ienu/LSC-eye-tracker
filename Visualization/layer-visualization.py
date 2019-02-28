# -*- coding: utf-8 -*-
#!/usr/bin/env python
import cv2
import os
import sys
import random
from keras import backend as K 
import tensorflow as tf
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt 
import PIL.Image as Image

def _get_data(path):
    """Function to load .npz file and get data from it randomly"""
    archive = np.load(path)
    images = archive['faceData']
    return images

if __name__ == '__main__':
    if len(sys.argv) < 3:
        """
            argv[1] path to the root directory of npz file
            argv[2] path to the root directory of model
        """
        print ("[ERROR] Please specify the path to npz file and the path to .h5 model file")
        exit(0)
    else:
        path_to_npz = sys.argv[1]
        path_to_model = sys.argv[2]

    
    for i in range(6):
        # Deal with 6 person's data
        if i==1:
            continue
        person = 'p0'+str(i)
        path_to_person_model = path_to_model + '/model_MPIIFaceGaze_' + person +'_cv_600.h5'
        path_to_person_npz   = path_to_npz + '/' + person + '/' +'data_MPIIFaceGaze_' + person + '_test.npz' 
        print ("Loading npz file: {}".format(path_to_person_npz))
        print ("Loading model file: {}".format(path_to_person_model))
        images = _get_data(path_to_person_npz)
        model = load_model(path_to_person_model)
        amount = images.shape[0]
        image_size = images.shape[1]
        index = 1337
        input_image = np.reshape(images[index], (-1, 224, 224, 3))


        # Using backend.function to get specific layers
        for index in range(4):
            # Visualize the output of layer 1-4
            layer = K.function([model.layers[0].input], [model.layers[index].output])
            f = layer([input_image])[0]

            # Tensor transpose for output
            output = np.transpose(f, (0, 3, 1, 2))

            # The first conv layer has 16 kernel
            image_map = Image.new('RGB', (image_size*4, image_size*4))
            for row in range(4):
                for col in range(4):
                    from_image = Image.fromarray((output[0][row*4+col]).astype('uint8')).convert('RGB')
                    image_map.paste(from_image, (col*image_size, row*image_size))
            path_to_save_image = './visualization/' + person + '/layer' + str(index) + '_output.png' 
            image_map.save(path_to_save_image)

