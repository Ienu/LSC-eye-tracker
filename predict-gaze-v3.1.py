# -*- coding: utf-8 -*-
'''
    Author: Wenyu
    Date: 2/12/2019
    Version: 3.1

    Function:
    v3.1: predict from the MPIIFaceGaze dataset
'''
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
import binascii
import time
import tensorflow as tf

import cv2

# TODO license required
# TODO: The sychronization remains problem

def gaze_estimate(tensor, width, height, scale):
    '''
        estimate the gaze point from the predicted tensor
    '''
    c = 1.0 / scale
    avg_x = 0
    avg_y = 0
    sum_p = 0
    for m in range(scale):
        for n in range(scale):
            x = (c / 2 + c * m + tensor[m][n][0]) * width
            y = (c / 2 + c * n + tensor[m][n][1]) * height
            p = tensor[m][n][2]
            #print(x, y, p)
            if p > 0:
                r = 1.0 / p
                if x >= 0 and x < width and y >= 0 and y < height:
                    avg_x += x * p
                    avg_y += + y * p
                    sum_p += p
    if sum_p > 0:
        avg_x /= sum_p
        avg_y /= sum_p
    return (avg_x, avg_y)


def main():
    file_name = 'p01.txt'

    amount = 0
    for index, line in enumerate(open(file_name,'r')):
        amount += 1

    print('amount = ', amount)

    # model = load_model(sys.argv[1])
    model = load_model('model_MPIIFaceGaze_p01_cv_last.h5')
    # the size of the screen
    width = 1280
    height = 800

    face_size = 224
    scale = 7

    index = 0

    # 加载图像
    with open(file_name, 'r') as file:
        for line in file.readlines():
            index += 1
            if index < int(amount * 9 / 10):
                continue
            vector = line.split(' ')
            # print(vector[0], vector[1], vector[2])
            image_src = cv2.imread(vector[0])
            # cv2.imshow('test', image_src)
            # cv2.waitKey(0)
            
            # TODO: according to YOLO v1, HSV color space need to be tested
            image_dst = cv2.resize(image_src, (face_size, face_size))
            image_src = cv2.resize(image_src, (width, height))

            x = np.expand_dims(image_dst, axis=0)
            x = x.astype('float32')
            x /= 255

            preds = model.predict(x)

            (px, py) = gaze_estimate(preds[0, :, :, :], width, height, scale)

            cv2.circle(image_src,
                       (width - int(vector[1]), int(vector[2])),
                       3, (0,0,255), 10)
            cv2.circle(image_src,
                       (width - int(px), int(py)),
                       3, (255,0,0), 5)
                           
            imageDst = cv2.resize(image_src, (1280, 720))
            cv2.imshow('dst', imageDst)
            key = cv2.waitKey(1000)
            if key == 27:
                break
                # TODO: should implement cursor control by python
                #f = open('D:\\mouse.txt', 'w')
                #f.write('%(x)d\t%(y)d' % {'x':(preds[0][0]) * 1600, \
                #                          'y':(preds[0][1]) * 900})
                #f.close()

if __name__ == '__main__':
    main()
