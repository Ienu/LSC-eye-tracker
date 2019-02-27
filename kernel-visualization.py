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

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std()+1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    # Convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ("[ERROR] Please specify the path to .h5 model file")
    path_to_model = sys.argv[1]
    model = load_model(path_to_model)

    # Visualize kernels in the first conv layer
    for i_kernel in range(16):
        input_image = model.input
        # Set a loss function, maximize the activation of kernels in the first layers 
        loss = K.mean(model.layers[0].output[:,:,:,i_kernel])
        # Compute the gradient of the input picture and its loss
        grads = K.gradients(loss, input_image)[0]
        # Normalize the gradients use L2
        # Add a 1e-5 constant to avoid Vanishing Gradient Problem
        grads /= (K.sqrt(K.mean(K.square(grads)))+1e-5)
        #This function return the gradients and loss of input image
        iterate = K.function([input_image, K.learning_phase()], [loss, grads])

        # Random seed
        np.random.seed(0)
        # Size of image 
        image_width = 224
        image_height = 224
        num_channels = 3
        # Generate some images with some noise and start with them
        # input_image_data = (255- np.random.randint(0,255,(1,  image_height, image_width, num_channels))) / 255.
        input_image_data = np.uint8(np.random.uniform(150, 180, (1, image_width, image_height, num_channels)))/255  # generate random image
        failed = False
        # Run gradient ascent
        print ("##########[Kernel_{}]##########".format(i_kernel))

        loss_value_pre = 0
        for i in range(500):
            loss_value, grads_value = iterate([input_image_data, 1])
            if i%100 == 0:
                print ("Iteration %d/%d, loss: %f"%(i, 500, loss_value))
                print ("Mean grad: %f"%(np.mean(grads_value)))
                # If the gradient vanished break the loop
                if all(np.abs(grads_val)<0.000001 for grads_val in grads_value.flatten()):
                    failed = True
                    print ("Failed")
                    break
                # If the loss value in current loop doesn't decrease, break the loop
                if loss_value_pre !=0 and loss_value_pre > loss_value:
                    break
                # Record loss value for each loop
                if loss_value_pre == 0:
                    loss_value_pre = loss_value

            input_image_data += grads_value*1
        plt.subplot(4,4, i_kernel+1)
        image_result = deprocess_image(input_image_data[0])
        image_result = np.reshape(image_result, (224,224, 3))
        plt.imshow (image_result)
    plt.show()
