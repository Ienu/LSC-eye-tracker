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

# TODO: The sychronization remains problem

# init
num_classes = 2

image_path = u'F:\\DataSet\\Test\\2\\image\\'

gazeData = np.loadtxt('F:\\DataSet\\Test\\2\\gazeData.txt')

model = load_model(sys.argv[1])

width = 1600
height = 900

scale = 7


# 加载图像
for index in range(40000, 50000):
    try:
        #time.sleep(0.1)
        # TODO: should implement real time without black boundary
        imgIdx = index + 0
        imageSrc = cv2.imread(image_path + '%d.bmp' % imgIdx)
        imageRs = cv2.resize(imageSrc, (224, 224))

        #cv2.imshow("rs", imageRs)
        #cv2.waitKey(10)
        
        #img = image.load_img(image_path, target_size=(224, 224))
    except IOError:
        print('cannot open')
        continue
    else:
        # 图像预处理
        
        try:
            #x = image.img_to_array(img)
            x = np.expand_dims(imageRs, axis=0)
            x = x.astype('float32')
            x /= 255

            #print(x.shape)

            preds = model.predict(x)
            #print(preds)

            imageSrc = cv2.resize(imageSrc, (width, height))

            c = 1.0 / scale
            avg_x = 0
            avg_y = 0
            sum_p = 0
            for m in range(scale):
                for n in range(scale):
                    x = (c / 2 + c * m + preds[0][m][n][0]) * width
                    y = (c / 2 + c * n + preds[0][m][n][1]) * height
                    p = preds[0][m][n][2]
                    #print(x, y, p)
                    if p > 0:
                        r = 1.0 / p
                        if x >= 0 and x < width and y >= 0 and y < height:
                            #cv2.circle(imageSrc, (width - int(x), \
                            #                     int(y)), int(r), \
                            #          (255,0,0), 1)
                            avg_x = avg_x + x * p
                            avg_y = avg_y + y * p
                            sum_p = sum_p + p

            cv2.circle(imageSrc, (width - int(gazeData[index][1]), \
                                  int(gazeData[index][2])), 3, \
                                  (0,0,255), 5)
            if sum_p > 0:
                cv2.circle(imageSrc, (width - int(avg_x / sum_p), \
                                      int(avg_y / sum_p)), 3, \
                           (255,0,0), 5)
                       
            imageDst = cv2.resize(imageSrc, (1280, 720))
            cv2.imshow('dst', imageDst)
            key = cv2.waitKey(10)
            if key == 27:
                break
            # TODO: should implement cursor control by python
            #f = open('D:\\mouse.txt', 'w')
            #f.write('%(x)d\t%(y)d' % {'x':(preds[0][0]) * 1600, \
            #                          'y':(preds[0][1]) * 900})
            #f.close()
        except TypeError:
            print('bmp error')
            continue
        else:
            continue


