# -*- coding: utf-8 -*-
'''
    Author: Wenyu
    Date: 2/11/2019
    Version: 4.3
    
    Function:
    v4.0: This script tries to process MPIIFaceGaze data into npz
    Then for our model test
    v4.1: use function to structured
    v4.2: Generate data for each person
    v4.3: Generate data from all person to one .npz file
    v5.0[insfan][4/25/2019]：Generate hole img data and face_roi data from all person to one .npz file
    v6.0[insfan][5/25/2019]:Generate face, face_mask, eyes, gaze_point data to .npz file
'''
import numpy as np
import cv2
import time
import os
import sys
from scipy.io import loadmat

# TODO: license required

# this code generate the whole dataset .npz file from the original images
# in folder 'image' with 'gazeData.txt', resize images to 224 * 224
# [insfan][5/25/2019] 
# TODO: a dataset info field should be added and save in the file as one field
# 

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

def generate_gaze_point(gaze_point, width, height):
    '''
        Normalize gaze point
    '''
    x = int(gaze_point[0])
    y = int(gaze_point[1])
    gaze_tensor = np.zeros([1, 1, 2], dtype='float32')
    gaze_tensor[0, 0, :] = [1.0*x / width, 1.0*y / height]
    return gaze_tensor

def generate_face_eyes_roi(face_landmarks, src_img):
    '''
        generate face and eyes roi from src img based face_landmarks
    '''
    # get face roi
    face_landmarks = [int(i) for i in face_landmarks]
    # cv2.circle(src_img, (face_landmarks[0], face_landmarks[1]), 2, [255, 0, 0])
    # cv2.imshow('landmarks', src_img)
    center_x = (face_landmarks[0] + face_landmarks[2] + face_landmarks[4] + face_landmarks[6] + face_landmarks[8] + face_landmarks[10]) / 6
    center_y = (face_landmarks[1] + face_landmarks[3] + face_landmarks[5] + face_landmarks[7] + face_landmarks[9] + face_landmarks[11]) / 6
    rect_w = int((center_x - face_landmarks[0]) * 1.5)
    # cv2.circle(src_img, (int(center_x), int(center_y)), 2, [0, 255, 0])
    top_l_x = int(center_x - rect_w)
    top_l_y = int(center_y - rect_w)
    down_r_x = int(center_x + rect_w)
    down_r_y = int(center_y + rect_w)
    if (down_r_x > src_img.shape[1]):
        down_r_x = src_img.shape[1]
    if (down_r_y > src_img.shape[0]):
        down_r_y = src_img.shape[0]
    if (top_l_x < 0):
        top_l_x = 0
    if (top_l_y < 0):
        top_l_y = 0
    face = src_img[top_l_y: down_r_y, top_l_x: down_r_x]

    # get face mask
    face_mask = np.zeros((src_img.shape[0], src_img.shape[1], 1), dtype=np.float32)
    # cv2.fillPoly(face_mask, np.array([[(top_l_y, top_l_x), (top_l_y, down_r_x), (down_r_y, down_r_x), (down_r_y, top_l_x)]], dtype=np.int32), color=1.0)
    cv2.fillPoly(face_mask,
                 np.array([[(top_l_x, top_l_y), (down_r_x, top_l_y), (down_r_x, down_r_y), (top_l_x, down_r_y)]],
                          dtype=np.int32), color=1.0)

    # get eyes roi
    eye_width = 2 * max(abs(face_landmarks[0] - face_landmarks[2]), abs(face_landmarks[4] - face_landmarks[6]))
    eye_height = 1.0 * eye_width
    eye_middle_right = ((face_landmarks[0] + face_landmarks[2])/2, (face_landmarks[1] + face_landmarks[3])/2)
    eye_middle_left = ((face_landmarks[4] + face_landmarks[6])/2, (face_landmarks[5] + face_landmarks[7])/2)
    top_l_x = abs(int(eye_middle_left[0] - eye_width / 2.0))
    top_l_y = abs(int(eye_middle_left[1] - eye_height / 2.0))
    down_r_x = abs(int(eye_middle_left[0] + eye_width / 2.0))
    down_r_y = abs(int(eye_middle_left[1] + eye_height / 2.0))
    # segment eye image
    eyes = []
    eyes.append(src_img[top_l_y: down_r_y, top_l_x: down_r_x])

    top_l_x = abs(int(eye_middle_right[0] - eye_width / 2.0))
    top_l_y = abs(int(eye_middle_right[1] - eye_height / 2.0))
    down_r_x = abs(int(eye_middle_right[0] + eye_width / 2.0))
    down_r_y = abs(int(eye_middle_right[1] + eye_height / 2.0))
    eyes.append(src_img[top_l_y: down_r_y, top_l_x: down_r_x])
    return face, face_mask, eyes

def vector_to_pitchyaw(vectors):
    r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.

    Args:
        vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
    """
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out
        
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
    mask_size = 36
    eyes_size = (60, 60)
    amount = 0
    index = 0
    save_name = 'MPIIFaceGaze_eyes_p00_50.npz'
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
        # [insfan][4/25/2019] one person example 
        break
    print ("total number of image is: %d"%(amount))
    # Create a numpy array to contain data  
    # mask_data = np.zeros([amount+1, mask_size, mask_size, 1], dtype='float32')
    # face_data = np.zeros([amount + 1, face_size, face_size, 3], dtype='uint8')
    amount = 50 # [insfan][4/25/2019] limit save number 
    left_eye_data = np.zeros([amount, eyes_size[0], eyes_size[1], 3], dtype='uint8')
    right_eye_data = np.zeros([amount, eyes_size[0], eyes_size[1], 3], dtype='uint8')
    gaze_data = np.zeros([amount, 1, 1, 2], dtype='float32')
    # scale = 7
    # eye_track_data = np.zeros([amount+1, scale, scale, 3], dtype='float32')
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
        print (file_name)
        with open(file_name, 'r') as file:
            for line in file.readlines():
                vector = line.split(' ')
                # print(vector[0], vector[1], vector[2])
                image_src = cv2.imread(vector[0])
                face, face_mask, eyes = generate_face_eyes_roi(vector[3:15], image_src)
                # cv2.imshow('image_src', image_src)
                # face_mask = cv2.resize(face_mask, (mask_size, mask_size))
                # face_mask = np.expand_dims(face_mask, axis=2)
                # cv2.imshow('face_mask', face_mask)
                # face = cv2.resize(face, (face_size, face_size))
                # cv2.imshow('face_roi', face)
                eyes[0] = cv2.resize(eyes[0], (eyes_size[1], eyes_size[0]))
                eyes[1] = cv2.resize(eyes[1], (eyes_size[1], eyes_size[0]))
                cv2.imshow('left eye', eyes[0])
                cv2.imshow('right eye', eyes[1])
                # eyes[0] = cv2.resize(eyes[0], (eyes_size[1], eyes_size[0]))
                # if cv2.waitKey(0) == 'q':
                #     exit(0)
                # TODO: according to YOLO v1, HSV color space need to be tested
                # image_dst = cv2.resize(image_src, (face_size, face_size))
                cv2.waitKey(1)
                # face_data[index, :, :, :] = face
                left_eye_data[index, :, :, :] = eyes[0]
                right_eye_data[index, :, :, :] = eyes[1]
                # print('mask shape: ', face_mask.shape)
                # mask_data[index, :, :, :] = face_mask
                gaze_data[index, :, :, :] = generate_gaze_point(vector[1:3], width, height)
                # pitchyam = vector_to_pitchyaw(vector[24:27])
                # TODO: according to YOLO v2, they used logistic activation to normalize
                # maybe [0, 1] is better than [-0.5, 0.5] according to Relu
                # eye_track_data[index, :, :, :] = generate_gaze_tensor(vector[1:3],
                #                                                     width,
                #                                                     height,
                #                                                     scale)
                index += 1
                # [insfan][5/25/2019] limit save number
                if (index >= amount):
                    break  
        print(index)
        # [insfan][4/25/2019] one person example
        break
    print('index', index)
    # TODO: a data info field should be saved within   
    # .npz file was saved in data folder     
    np.savez_compressed(dataset_folder + '/' + save_name,
                       # faceData=face_data,
                       leftEyeData=left_eye_data,
                       rightEyeData=right_eye_data,
                       # faceMaskData=mask_data,
                       gazeData=gaze_data)

if __name__ == '__main__':
    main()
