import numpy as np
import cv2
import time
import os

'''
    Author: Wenyu
    Date: 2/11/2019
    Version: 4.0
    
    Function:
    v4.0: This script tries to process MPIIFaceGaze data into npz
    Then for our model test
'''

# TODO: coding type like UTF-8 should be added
# TODO: license required

# this code generate the whole dataset .npz file from the original images
# in folder 'image' with 'gazeData.txt', resize images to 224 * 224

# TODO: a dataset info field should be added and save in the file as one field

# Warning
print('################################################################')
print('# WARNING: The Code Should Be Tested On A Small Dataset First! #')
print('################################################################')

# TODO: implement input argument like argv[1] with default value
saveName = 'data_MPIIFaceGaze_p00_test.npz'

faceSize = 224

# The width and height values are recorded in the calibration data .mat
width = 1280
height = 800

file_name = 'p00.txt'

amount = 0
for index, line in enumerate(open(file_name,'r')):
    amount += 1

print('amount = ', amount)

# we use 'uint8' to reduce file size for storage, but require 'float32'
# when computing on GPU
faceData = np.zeros([amount, faceSize, faceSize, 3], dtype='uint8')
# this generates a mapping tensor for regression from the original
# tensor region size
scale = 7
# dim-4 indicates (x, y, p) 
eyeTrackData = np.zeros([amount, scale, scale, 3], dtype='float32')

t_start = time.time()

index = 0

with open(file_name, 'r') as file:
    for line in file.readlines():
        vector = line.split(' ')
        # print(vector[0], vector[1], vector[2])
        image_src = cv2.imread(vector[0])
        # cv2.imshow('test', image_src)
        # cv2.waitKey(0)
        # TODO: according to YOLO v1, HSV color space need to be tested
        image_dst = cv2.resize(image_src, (faceSize, faceSize))

        faceData[index, :, :, :] = image_dst
        # TODO: according to YOLO v2, they used logistic activation to normalize
        # maybe [0, 1] is better than [-0.5, 0.5] according to Relu
        w = int(vector[1])
        h = int(vector[2])
        c = 1.0 / scale
        for m in range(scale):
            for n in range(scale):
                x = (c / 2 + c * m) * width
                y = (c / 2 + c * n) * height
                # the distance from the ground truth to the point of each cell
                # predicts
                dis = np.sqrt((x - w) * (x - w) + (y - h) * (y - h))
                eyeTrackData[index, m, n, :] = [(w - x) / width, \
                                            (h - y) / height, \
                                            1.0 / (dis + 1.0)]

        print(vector[0] + ': %f, %f, %f, %f, %f' % (w, h, x, y, dis))
        index += 1




##
##gazeData = np.loadtxt('gazeData.txt')
##
### TODO: these parameters can set as input arguments with default values
##width = 1600
##height = 900
##amount = 40000
##
### TODO: should pre-calculate memory needed with type 'float32'
### YOLO v2 use 448 * 448, we need to consider large data training strategy
##faceSize = 224
##
### we use 'uint8' to reduce file size for storage, but require 'float32'
### when computing on GPU
##faceData = np.zeros([amount, faceSize, faceSize, 3], dtype='uint8')
### this generates a mapping tensor for regression from the original
### tensor region size
##scale = 7
### dim-4 indicates (x, y, p) 
##eyeTrackData = np.zeros([amount, scale, scale, 3], dtype='float32')
##
##t_start = time.time()
##
##for i in range(amount):
##    # TODO: implement sample non-exist process
##    # TODO: (0, 0) gaze point need to be removed
##    fileName = u'image\\%d.bmp' % i
##    imageSrc = cv2.imread(fileName)
##    # TODO: according to YOLO v1, HSV color space need to be tested
##    imageDst = cv2.resize(imageSrc, (faceSize, faceSize))
##    faceData[i, :, :, :] = imageDst
##    # TODO: according to YOLO v2, they used logistic activation to normalize
##    # maybe [0, 1] is better than [-0.5, 0.5] according to Relu
##    w = gazeData[i][1]
##    h = gazeData[i][2]
##    c = 1.0 / scale
##    for m in range(scale):
##        for n in range(scale):
##            x = (c / 2 + c * m) * width
##            y = (c / 2 + c * n) * height
##            # the distance from the ground truth to the point of each cell
##            # predicts
##            dis = np.sqrt((x - w) * (x - w) + (y - h) * (y - h))
##            eyeTrackData[i, m, n, :] = [(w - x) / width, \
##                                        (h - y) / height, \
##                                        1.0 / (dis + 1.0)]
##
##    print(fileName + ': %f, %f, %f, %f, %f' % (w, h, x, y, dis))
##

        
# TODO: a data info field should be saved within        
np.savez_compressed(saveName, faceData=faceData, eyeTrackData=eyeTrackData)

print(time.time()-t_start)
