"""Utility methods for generating gazemaps."""
"""v0.0[insfan][5/25/2019] gengerate gazemaps from Utility datasets"""
import cv2 as cv
import numpy as np

height_to_eyeball_radius_ratio = 1.1          # assume value,  paper is 1.2
eyeball_radius_to_iris_diameter_ratio = 1.0   # estimate value

def from_gaze2d(landmarks, intput_size, output_size, scale=1.0):
    """Generate a normalized pictorial representation of 3D gaze direction."""
    gazemaps = []
    oh, ow = np.round(scale * np.asarray(output_size)).astype(np.int32)

    # Draw iris
    gazemap = np.zeros((intput_size[0], intput_size[1]), dtype=np.float32)
    cv.fillPoly(gazemap, np.array([landmarks[16:48, :]], int), color=1.0)
    gazemap = cv.resize(gazemap, (ow, oh))
    gazemaps.append(gazemap)

    # Draw interior margin
    gazemap = np.zeros((intput_size[0], intput_size[1]), dtype=np.float32)
    cv.fillPoly(gazemap, np.array([landmarks[0:16, :]], int), color=1.0)
    gazemap = cv.resize(gazemap, (ow, oh))
    gazemaps.append(gazemap)

    return np.asarray(gazemaps)
