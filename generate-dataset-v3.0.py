import numpy as np
import cv2
import time

# TODO: code info and author, modified date should be added
# TODO: coding type like UTF-8 should be added

# this code generate the whole dataset .npz file from the original images
# in folder 'image' with 'gazeData.txt', resize images to 224 * 224

# TODO: a dataset info field should be added and save in the file as one field

# Warning
print('################################################################')
print('# WARNING: The Code Should Be Tested On A Small Dataset First! #')
print('################################################################')

# TODO: implement input argument like argv[1] with default value
saveName = 'data_3v.npz'

gazeData = np.loadtxt('gazeData.txt')

# TODO: these parameters can set as input arguments with default values
width = 1600
height = 900
amount = 40000

# TODO: should pre-calculate memory needed with type 'float32'
# YOLO v2 use 448 * 448, we need to consider large data training strategy
faceSize = 224

# we use 'uint8' to reduce file size for storage, but require 'float32'
# when computing on GPU
faceData = np.zeros([amount, faceSize, faceSize, 3], dtype='uint8')
# this generates a mapping tensor for regression from the original
# tensor region size
scale = 7
# dim-4 indicates (x, y, p) 
eyeTrackData = np.zeros([amount, scale, scale, 3], dtype='float32')

t_start = time.time()

for i in range(amount):
    # TODO: implement sample non-exist process
    # TODO: (0, 0) gaze point need to be removed
    fileName = u'image\\%d.bmp' % i
    imageSrc = cv2.imread(fileName)
    # TODO: according to YOLO v1, HSV color space need to be tested
    imageDst = cv2.resize(imageSrc, (faceSize, faceSize))
    faceData[i, :, :, :] = imageDst
    # TODO: according to YOLO v2, they used logistic activation to normalize
    # maybe [0, 1] is better than [-0.5, 0.5] according to Relu
    w = gazeData[i][1]
    h = gazeData[i][2]
    c = 1.0 / scale
    for m in range(scale):
        for n in range(scale):
            x = (c / 2 + c * m) * width
            y = (c / 2 + c * n) * height
            # the distance from the ground truth to the point of each cell
            # predicts
            dis = np.sqrt((x - w) * (x - w) + (y - h) * (y - h))
            eyeTrackData[i, m, n, :] = [(w - x) / width, \
                                        (h - y) / height, \
                                        1.0 / (dis + 1.0)]

    print(fileName + ': %f, %f, %f, %f, %f' % (w, h, x, y, dis))

# TODO: a data info field should be saved within        
np.savez_compressed(saveName, faceData=faceData, eyeTrackData=eyeTrackData)

print(time.time()-t_start)
