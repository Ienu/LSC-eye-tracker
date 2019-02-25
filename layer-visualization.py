# -*- coding: utf-8 -*-
#!/usr/bin/env python
import cv2
import os
import sys
import random
from keras import backend as K 
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
        print ("[ERROR] Please specify the path to npz file and the path to .h5 model file")
        exit(0)
    else:
        path_to_npz = sys.argv[1]
        images = _get_data(path_to_npz)
        path_to_model = sys.argv[2]
    model = load_model(path_to_model)

    amount = images.shape[0]
    image_size = images.shape[1]
    index = 1337
    input_image = np.reshape(images[index], (-1, 224, 224, 3))

    # Using backend.function to get specific layers
    print (len(model.layers))
    layer_0 = K.function([model.layers[0].input], [model.layers[0].output])
    f_0 = layer_0([input_image])[0]

    # Tensor transpose for output
    output = np.transpose(f_0, (0, 3, 1, 2))

    # The first conv layer has 16 kernel
    image_map = Image.new('RGB', (image_size*4, image_size*4))
    for row in range(4):
        for col in range(4):
            from_image = Image.fromarray((output[0][row*4+col]).astype('uint8')).convert('RGB')
            image_map.paste(from_image, (col*image_size, row*image_size))
    image_map.save('./visualization/layer0_output.png')

