import urllib.request
import urllib.parse

import bz2
import dlib
import shutil
import os
import cv2 as cv
import numpy as np

_face_detector = None
_landmarks_predictor = None

def _get_dlib_data_file(dat_name):
    dat_dir = os.path.relpath('%s/../3rdparty' % os.path.basename(__file__))
    dat_path = '%s/%s' % (dat_dir, dat_name)
    if not os.path.isdir(dat_dir):
        os.mkdir(dat_dir)

    # Download trained shape detector
    if not os.path.isfile(dat_path):
        with urllib.request.urlopen('http://dlib.net/files/%s.bz2' % dat_name) as response:
            with bz2.BZ2File(response) as bzf, open(dat_path, 'wb') as f:
                shutil.copyfileobj(bzf, f)

    return dat_path

def _get_opencv_xml(xml_name):
    xml_dir = os.path.relpath('%s/../3rdparty' % os.path.basename(__file__))
    xml_path = '%s/%s' % (xml_dir, xml_name)
    if not os.path.isdir(xml_dir):
        os.mkdir(xml_dir)

    # Download trained shape detector
    if not os.path.isfile(xml_path):
        url_stem = 'https://raw.githubusercontent.com/opencv/opencv/master/data/lbpcascades'
        with urllib.request.urlopen('%s/%s' % (url_stem, xml_name)) as response:
            with open(xml_path, 'wb') as f:
                shutil.copyfileobj(response, f)

    return xml_path

def get_landmarks_predictor():
    """Get a singleton dlib face landmark predictor."""
    global _landmarks_predictor
    if not _landmarks_predictor:
        dat_path = _get_dlib_data_file('shape_predictor_5_face_landmarks.dat')
        # dat_path = _get_dlib_data_file('shape_predictor_68_face_landmarks.dat')
        _landmarks_predictor = dlib.shape_predictor(dat_path)
    return _landmarks_predictor

def get_face_detector():
    """Get a singleton dlib face detector."""
    global _face_detector
    if not _face_detector:
        try:
            dat_path = _get_dlib_data_file('mmod_human_face_detector.dat')
            _face_detector = dlib.cnn_face_detection_model_v1(dat_path)
        except:
            xml_path = _get_opencv_xml('lbpcascade_frontalface_improved.xml')
            _face_detector = cv.CascadeClassifier(xml_path)
    return _face_detector

def detect_landmarks(grey, faces):
    """Detect 5-point facial landmarks for faces in frame."""
    predictor = get_landmarks_predictor()
    landmarks = []
    for face in faces:
        l, t, w, h = face
        rectangle = dlib.rectangle(left=int(l), top=int(t), right=int(l+w), bottom=int(t+h))
        landmarks_dlib = predictor(grey, rectangle)

        def tuple_from_dlib_shape(index):
            p = landmarks_dlib.part(index)
            return (p.x, p.y)

        num_landmarks = landmarks_dlib.num_parts
        landmarks.append(np.array([tuple_from_dlib_shape(i) for i in range(num_landmarks)]))
    return landmarks

def segment_eyes(grey, faces, landmarks):
        """From found landmarks in previous steps, segment eye image."""
        eyes = []

        # Final output dimensions
        oh, ow = (108, 180)

        # Select which landmarks (raw/smoothed) to use
        frame_landmarks = landmarks

        for face, landmarks in zip(faces, frame_landmarks):
            # Segment eyes
            # for corner1, corner2, is_left in [(36, 39, True), (42, 45, False)]:
            for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
                x1, y1 = landmarks[corner1, :]
                x2, y2 = landmarks[corner2, :]
                eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
                if eye_width == 0.0:
                    continue
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

                # Centre image on middle of eye
                translate_mat = np.asmatrix(np.eye(3))
                translate_mat[:2, 2] = [[-cx], [-cy]]
                inv_translate_mat = np.asmatrix(np.eye(3))
                inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

                # Rotate to be upright
                roll = 0.0 if x1 == x2 else np.arctan((y2 - y1) / (x2 - x1))
                rotate_mat = np.asmatrix(np.eye(3))
                cos = np.cos(-roll)
                sin = np.sin(-roll)
                rotate_mat[0, 0] = cos
                rotate_mat[0, 1] = -sin
                rotate_mat[1, 0] = sin
                rotate_mat[1, 1] = cos
                inv_rotate_mat = rotate_mat.T

                # Scale
                scale = ow / eye_width
                scale_mat = np.asmatrix(np.eye(3))
                scale_mat[0, 0] = scale_mat[1, 1] = scale
                inv_scale = 1.0 / scale
                inv_scale_mat = np.asmatrix(np.eye(3))
                inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

                # Centre image
                centre_mat = np.asmatrix(np.eye(3))
                centre_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
                inv_centre_mat = np.asmatrix(np.eye(3))
                inv_centre_mat[:2, 2] = -centre_mat[:2, 2]

                # Get rotated and scaled, and segmented image
                transform_mat = centre_mat * scale_mat * rotate_mat * translate_mat
                inv_transform_mat = (inv_translate_mat * inv_rotate_mat * inv_scale_mat *
                                     inv_centre_mat)
                eye_image = cv.warpAffine(grey, transform_mat[:2, :], (ow, oh))
                if is_left:
                    eye_image = np.fliplr(eye_image)
                eyes.append({
                    'image': eye_image,
                    'inv_landmarks_transform_mat': inv_transform_mat,
                    'side': 'left' if is_left else 'right',
                })
        return eyes


#path = input("Image Path: ")
path = './'
#capture = cv.VideoCapture(0)
#capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
#capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

for files in os.listdir(path):
    if os.path.splitext(files)[1] == '.jpg' or True:
        #ret, bgr = capture.read()
        bgr = cv.imread('demo.jpg')
        bgr = cv.flip(bgr, flipCode=1)  # Mirror
        #bgr = cv.imread(files)
        grey = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

        detector = get_face_detector()
        if detector.__class__.__name__ == 'CascadeClassifier':
            detections = detector.detectMultiScale(grey)
        else:
            detections = detector(cv.resize(grey, (0, 0), fx=0.5, fy=0.5), 0)

        faces = []
        for d in detections:
            try:
                l, t, r, b = d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()
                l *= 2
                t *= 2
                r *= 2
                b *= 2
                w, h = r - l, b - t
            except AttributeError:  # Using OpenCV LBP detector on CPU
                l, t, w, h = d
            faces.append((l, t, w, h))
        faces.sort(key=lambda bbox: bbox[0])
        landmarks = detect_landmarks(grey, faces)
        for f, face in enumerate(faces):
            for landmark in landmarks[f][:-1]:
                cv.drawMarker(bgr, tuple(np.round(landmark).astype(np.int32)),
                color=(0, 0, 255), markerType=cv.MARKER_STAR,
                markerSize=2, thickness=1, line_type=cv.LINE_AA)

            cv.rectangle(bgr, tuple(np.round(face[:2]).astype(np.int32)), 
            tuple(np.round(np.add(face[:2], face[2:])).astype(np.int32)), 
            color=(0, 255, 255), thickness=1, lineType=cv.LINE_AA,)

        eyes = segment_eyes(grey, faces, landmarks)

        eye_image_left = eyes[0]['image']
        eye_image_right = eyes[1]['image']
        cv.imshow('left', eye_image_left)
        cv.imshow('right', eye_image_right)
        cv.imshow('faces', bgr)
        if cv.waitKey(0) == 'q':
            break


