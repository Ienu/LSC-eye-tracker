# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import h5py
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

import os
from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model

from keras import backend as K
import utils

def conv_output(model, layer_name, img):
    """Get the output of conv layer.

    Args:
           model: keras model.
           layer_name: name of layer in the model.
           img: processed input image.

    Returns:
           intermediate_output: feature map.
    """
    # this is the placeholder for the input images
    input_img = model.input

    try:
        # this is the placeholder for the conv output
        out_conv = model.get_layer(layer_name).output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    # get the intermediate layer model
    intermediate_layer_model = Model(inputs=input_img, outputs=out_conv)

    # get the output of intermediate layer model
    intermediate_output = intermediate_layer_model.predict(img)

    return intermediate_output

def conv_filter(model, layer_name, img):
    """Get the filter of conv layer.

    Args:
           model: keras model.
           layer_name: name of layer in the model.
           img: processed input image.

    Returns:
           filters.
    """
    # this is the placeholder for the input images
    input_img = model.input

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    try:
        layer_output = layer_dict[layer_name].output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    kept_filters = []
    for i in range(layer_output.shape[-1]):
        loss = K.mean(layer_output[:, :, :, i])

        # compute the gradient of the input picture with this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = utils.normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.
        # run gradient ascent for 20 steps
        fimg = img.copy()

        for j in range(40):
            loss_value, grads_value = iterate([fimg])
            fimg += grads_value * step

        # decode the resulting input image
        fimg = utils.deprocess_image(fimg[0])
        kept_filters.append((fimg, loss_value))

        # sort filter result
        kept_filters.sort(key=lambda x: x[1], reverse=True)

    return np.array([f[0] for f in kept_filters])


# init
num_classes = 2

print(sys.argv[1])
f = h5py.File(sys.argv[1])

x = np.array(f['faceData'])
y = np.array(f['eyeTrackData'])

x = np.transpose(x, (3, 2, 1, 0))
y = np.transpose(y, (1, 0))

print('x shape: ', x.shape)
print('y shape: ', y.shape)

x = x.astype('float32')
x /= 255
y = y.astype('float32')

model = load_model(sys.argv[2])

file = open(sys.argv[3], "w")

# model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(num_classes))

model.summary()

# deconv
#out = conv_output(model, 'activation_2', x)
#print(out.shape)
#m_out = np.mean(out, 0)
#m_out = out[0]
#for k in range(32):
#	plt.subplot(4, 8, k + 1)
#	plt.imshow(m_out[:, :, k])
#	plt.axis('off')
#plt.show()
	
out = conv_filter(model, 'activation_1', x)
print(out.shape)



















