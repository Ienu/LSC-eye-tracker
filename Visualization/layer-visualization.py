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

def resize_image(image):
    height = width = 224
    top, bottom, left, right = (0,0,0,0)
    h,w,c = image.shape
    longest_edge = max(h,w)

    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass 
    
    #RGB颜色
    BLACK = [0, 0, 0]
    
    #给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    
    #调整图像大小并返回
    return cv2.resize(constant, (height, width))

def _get_conv_weight(model, kernel_size, layer_name):
    weights = model.get_layer(layer_name).get_weights()
    weights = np.asarray(weights[0])
    mean_weights = np.mean(weights, axis=0)
    mean_weights = np.reshape(mean_weights, (3,3,1,kernel_size))
    return mean_weights

def image_deconv(image, Filter):
    image = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC)
    image = np.reshape(image, (-1,224,224,1))
    image_tensor = tf.convert_to_tensor(image)
    Filter = np.reshape(Filter, (3,3,1,1))
    Filter = np.transpose(Filter, (1,0,2,3))
    Filter = tf.convert_to_tensor(Filter)
    deconv = tf.nn.conv2d_transpose(image_tensor, Filter, [1,224,224,1], strides=[1,1,1,1], padding='SAME')
    deconv_image = tf.Session().run(deconv)
    deconv_image = np.reshape(deconv_image, (224,224))
    return deconv_image 
    

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
    
    net_layers = [
        'conv16', 'leReLU1', 'MaxPool1', 'BN1', 
        'conv32', 'leReLU2', 'MaxPool2', 'BN2',
        'conv64', 'leReLU3', 'MaxPool3', 'BN3',
        'conv128', 'leReLU4', 'MaxPool4', 'BN4',  
        'conv256', 'leReLU5', 'MaxPool5', 'BN5',
        'conv512', 'leReLU6', 'MaxPool6', 'BN6',
        'conv1024', 'leReLU7', 'BN7',
        'conv256', 'leReLU8', 'BN8',
        'conv512', 'leReLU9', 'BN9']

    
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
        image = np.reshape(images[index], (224,224,3))
        image = image[54:222, 67:161, :]
        #image = resize_image(image)
        image = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('1337', image)
        cv2.waitKey()
        input_image = np.reshape(image, (-1, 224, 224, 3))

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
                    image = cv2.resize(output[0][row*4+col], (224,224), interpolation=cv2.INTER_CUBIC)
                    from_image = Image.fromarray(image.astype('uint8')).convert('RGB')
                    image_map.paste(from_image, (col*image_size, row*image_size))
            path_to_save_image = './visualization/' + person + '/layer_' + str(net_layers[index]) + '_output.png' 
            image_map.save(path_to_save_image)

            deconv_image_map = Image.new('RGB', (image_size*4, image_size*4))
            Filters = _get_conv_weight(model, 16, 'conv2d_1')
            for row in range(4):
                for col in range(4):
                    image = output[0][row*4+col]
                    Filter = Filters[:,:,:,row*4+col]
                    deconv_image = image_deconv(image, Filter)
                    from_image = Image.fromarray(deconv_image.astype('uint8'))
                    image_map.paste(from_image, (col*image_size, row*image_size))
            path_to_save_image = './visualization/' + person + '/layer_' + str(net_layers[index]) + '_deconv_output.png'
            deconv_image_map.save(path_to_save_image)

        for index in range(4):
            layer = K.function([model.layers[0].input], [model.layers[index+4].output])
            f = layer([input_image])[0]

            output = np.transpose(f, (0,3,1,2))
            image_map = Image.new('RGB', (image_size*8, image_size*4))
            for row in range(4):
                for col in range(8):
                    image = cv2.resize(output[0][row*4+col], (224,224), interpolation=cv2.INTER_CUBIC)
                    from_image = Image.fromarray(image.astype('uint8')).convert('RGB')
                    image_map.paste(from_image, (col*image_size, row*image_size))
            path_to_save_image = './visualization/' + person + '/layer' + str(net_layers[index+4]) + '_output.png'
            image_map.save(path_to_save_image)

        for index in range(4):
            layer = K.function([model.layers[0].input], [model.layers[index+4*2].output])
            f = layer([input_image])[0]

            output = np.transpose(f, (0,3,1,2))
            image_map = Image.new('RGB', (image_size*8, image_size*8))
            for row in range(8):
                for col in range(8):
                    image = cv2.resize(output[0][row*4+col], (224,224), interpolation=cv2.INTER_CUBIC)
                    from_image = Image.fromarray(image.astype('uint8')).convert('RGB')
                    image_map.paste(from_image, (col*image_size, row*image_size))
            path_to_save_image = './visualization/' + person + '/layer' + str(net_layers[index+4*2]) + '_output.png'
            image_map.save(path_to_save_image)

        for index in range(4):
            layer = K.function([model.layers[0].input], [model.layers[index+4*3].output])
            f = layer([input_image])[0]

            output = np.transpose(f, (0,3,1,2))
            image_map = Image.new('RGB', (image_size*16, image_size*8))
            for row in range(8):
                for col in range(16):
                    image = cv2.resize(output[0][row*4+col], (224,224), interpolation=cv2.INTER_CUBIC)
                    from_image = Image.fromarray(image.astype('uint8')).convert('RGB')
                    image_map.paste(from_image, (col*image_size, row*image_size))
            path_to_save_image = './visualization/' + person + '/layer' + str(net_layers[index+4*3]) + '_output.png'
            image_map.save(path_to_save_image)
        
        for index in range(4):
            layer = K.function([model.layers[0].input], [model.layers[index+4*4].output])
            f = layer([input_image])[0]

            output = np.transpose(f, (0,3,1,2))
            image_map = Image.new('RGB', (image_size*16, image_size*16))
            for row in range(16):
                for col in range(16):
                    image = cv2.resize(output[0][row*4+col], (224,224), interpolation=cv2.INTER_CUBIC)
                    from_image = Image.fromarray(image.astype('uint8')).convert('RGB')
                    image_map.paste(from_image, (col*image_size, row*image_size))
            path_to_save_image = './visualization/' + person + '/layer' + str(net_layers[index+4*4]) + '_output.png'
            image_map.save(path_to_save_image)

        for index in range(4):
            layer = K.function([model.layers[0].input], [model.layers[index+4*5].output])
            f = layer([input_image])[0]

            output = np.transpose(f, (0,3,1,2))
            image_map = Image.new('RGB', (image_size*32, image_size*16))
            for row in range(16):
                for col in range(32):
                    image = cv2.resize(output[0][row*4+col], (224,224), interpolation=cv2.INTER_CUBIC)
                    from_image = Image.fromarray(image.astype('uint8')).convert('RGB')
                    image_map.paste(from_image, (col*image_size, row*image_size))
            path_to_save_image = './visualization/' + person + '/layer' + str(net_layers[index+4*5]) + '_output.png'
            image_map.save(path_to_save_image)
        
        for index in range(3):
            layer = K.function([model.layers[0].input], [model.layers[index+4*6].output])
            f = layer([input_image])[0]

            output = np.transpose(f, (0,3,1,2))
            image_map = Image.new('RGB', (image_size*32, image_size*32))
            for row in range(32):
                for col in range(32):
                    image = cv2.resize(output[0][row*4+col], (224,224), interpolation=cv2.INTER_CUBIC)
                    from_image = Image.fromarray(image.astype('uint8')).convert('RGB')
                    image_map.paste(from_image, (col*image_size, row*image_size))
            path_to_save_image = './visualization/' + person + '/layer' + str(net_layers[index+4*6]) + '_output.png'
            image_map.save(path_to_save_image)

        for index in range(3):
            layer = K.function([model.layers[0].input], [model.layers[index+4*6+3].output])
            f = layer([input_image])[0]

            output = np.transpose(f, (0,3,1,2))
            image_map = Image.new('RGB', (image_size*16, image_size*16))
            for row in range(16):
                for col in range(16):
                    image = cv2.resize(output[0][row*4+col], (224,224), interpolation=cv2.INTER_CUBIC)
                    from_image = Image.fromarray(image.astype('uint8')).convert('RGB')
                    image_map.paste(from_image, (col*image_size, row*image_size))
            path_to_save_image = './visualization/' + person + '/layer' + str(net_layers[index+4*6+3]) + '_output.png'
            image_map.save(path_to_save_image)
        
        for index in range(3):
            layer = K.function([model.layers[0].input], [model.layers[index+4*6+3*2].output])
            f = layer([input_image])[0]

            output = np.transpose(f, (0,3,1,2))
            image_map = Image.new('RGB', (image_size*32, image_size*16))
            for row in range(16):
                for col in range(32):
                    image = cv2.resize(output[0][row*4+col], (224,224), interpolation=cv2.INTER_CUBIC)
                    from_image = Image.fromarray(image.astype('uint8')).convert('RGB')
                    image_map.paste(from_image, (col*image_size, row*image_size))
            path_to_save_image = './visualization/' + person + '/layer' + str(net_layers[index+4*6+3*2]) + '_output.png'
            image_map.save(path_to_save_image)
        
        
            


