# -*- coding: utf-8 -*-
'''
    Author: Wenyu
    Date: 2/11/2019
    Version: generate-data-v4.3
    
    Function:
    v1.0[insfan][4/24/2019]: This script tries to process MPIIFaceGaze data gaze point distribution
'''
import numpy as np
import cv2
import time
import os
import sys
from scipy.io import loadmat
import random
# TODO: license required

# this code generate the whole dataset .npz file from the original images
# in folder 'image' with 'gazeData.txt', resize images to 224 * 224

# TODO: a dataset info field should be added and save in the file as one field

DATASET_ROOT = ''

def load_mat(path):
    m = loadmat(path)
    print ("Screen height: {}".format(m['height_pixel']))
    print ("Screen width: {}".format(m['width_pixel']))
    return m['height_pixel'], m['width_pixel']


def generate_gaze_tensor(gaze_point, width, height, scale):
    '''
        generate gaze tensor from gaze point
    '''
    w = int(gaze_point[0])
    h = int(gaze_point[1])
    c = 1.0 / scale
    gaze_tensor = np.zeros([scale, scale, 3], dtype='float32')
    for m in range(scale):
        for n in range(scale):
            x = (c / 2 + c * m) * width
            y = (c / 2 + c * n) * height
            # the distance from the gaze point to the point of each cell predicts
            dis = np.sqrt((x - w) * (x - w) + (y - h) * (y - h))
            gaze_tensor[m, n, :] = [(w - x) / width,
                                    (h - y) / height,
                                    1.0 / (dis + 1.0)]
    return gaze_tensor
def generate_face_roi(face_landmarks, src_img):
    '''
        generate face roi from src img based face_landmarks
    '''
    face_landmarks = [int(i) for i in face_landmarks]
    cv2.circle(src_img, (face_landmarks[0], face_landmarks[1]), 2, [255, 0, 0])
    # cv2.imshow('landmarks', src_img)
    # print("0x: %d")
    # print(face_landmarks[0].dtype)
    center_x = (face_landmarks[0] + face_landmarks[2] + face_landmarks[4] + face_landmarks[6] + face_landmarks[8] + face_landmarks[10]) / 6
    center_y = (face_landmarks[1] + face_landmarks[3] + face_landmarks[5] + face_landmarks[7] + face_landmarks[9] + face_landmarks[11]) / 6
    rect_w = int((center_x - face_landmarks[0]) * 1.5)
    cv2.circle(src_img, (int(center_x), int(center_y)), 2, [0, 255, 0])
    top_l_x = int(center_x - rect_w)
    top_l_y = int(center_y - rect_w)
    down_r_x = int(center_x + rect_w)
    down_r_y = int(center_y + rect_w)
    cv2.imshow('landmarks', src_img[top_l_y: down_r_y, top_l_x: down_r_x])
    # return src_img[top_l_y: down_r_y, top_l_x: down_r_x]
    return src_img
        
def main():
    """
        main function
        argv[1]: dataset root folder
    """

    # change current dir to dataset folder
    dataset_folder = sys.argv[1]
    # Warning
    print('################################################################')
    print('# WARNING: The Code Should Be Tested On A Small Dataset First! #')
    print('#            THIS code should run in dataset folder!           #')
    print('################################################################')

    face_size = 224
    amount = 0
    index = 0
    distribute_map = np.zeros((2000, 2000, 3), np.uint8)
    save_name = 'data_MPIIFaceGaze_all_test.npz'
    for person in os.listdir(dataset_folder):
        """Get the total number of all data"""

        # Change current dir to person dir
        # os.chdir(os.path.abspath(os.path.join(dataset_folder, person)))
        # Determine if it is a folder
        if (os.path.isfile(os.path.abspath(os.path.join(dataset_folder, person)))):   #
            continue
        os.chdir(os.path.abspath(os.path.join(dataset_folder, person)))
        file_name = person + '.txt'
        print (file_name)
        for line in enumerate(open(file_name,'r')):
            amount += 1
        # one person example
        break
    print ("total number of image is: %d"%(amount))
    # Create a numpy array to contain data  
    face_data = np.zeros([amount+1, face_size, face_size, 3], dtype='uint8')
    scale = 7
    eye_track_data = np.zeros([amount+1, scale, scale, 3], dtype='float32')
    for person in os.listdir(dataset_folder):
        # Determine if it is a folder
        if (os.path.isfile(os.path.abspath(os.path.join(dataset_folder, person))) ):  #
            continue
        path_to_mat_file = os.path.abspath(os.path.join(dataset_folder, person+'/Calibration/screenSize.mat'))
        # The width and height values are recorded in the calibration data .mat
        height, width = load_mat(path_to_mat_file)
        # Change current dir to person dir
        os.chdir(os.path.abspath(os.path.join(dataset_folder, person)))
        file_name = person + '.txt'
        bgr = [random.randint(0, 255) for i in range(3)]
        with open(file_name, 'r') as file:
            for line in file.readlines():
                vector = line.split(' ')
                # print(vector[0], vector[1], vector[2])
                # image_src = cv2.imread(vector[0])
                # face_roi = generate_face_roi(vector[3:15], image_src)
                # cv2.imshow('test', face_roi)
                # cv2.waitKey(0)
                # TODO: according to YOLO v1, HSV color space need to be tested
                # image_dst = cv2.resize(image_src, (face_size, face_size))
		
                # face_data[index, :, :, :] = image_dst
                # TODO: according to YOLO v2, they used logistic activation to normalize
                # maybe [0, 1] is better than [-0.5, 0.5] according to Relu
                # eye_track_data[index, :, :, :] = generate_gaze_tensor(vector[1:3],
                #                                                     width,
                #                                                     height,
                #                                                     scale)
                cv2.circle(distribute_map, (int(vector[1]), int(vector[2])), 2, bgr)
                index += 1


        print(index)
        # one person example
        # break
    print('index', index)
    cv2.imshow('distribute_map', distribute_map)
    cv2.waitKey(0)
    # TODO: a data info field should be saved within   
    # .npz file was saved in data folder     
    # np.savez_compressed(dataset_folder + '/' + save_name,
    #                    faceData=face_data,
    #                    eyeTrackData=eye_track_data)

if __name__ == '__main__':
    main()
