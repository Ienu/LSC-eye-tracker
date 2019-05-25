"""HDF5 data source for gaze estimation."""
"""v0.0[insfan][5/25/2019] Preprocessing datasets from mp2 npz file"""
from threading import Lock
from typing import List

import cv2 as cv
import h5py
import numpy as np
import tensorflow as tf

from core import BaseDataSource
import util.gazemap                       # file route may be a error


class NPZSource(BaseDataSource):
    """HDF5 data loading class (using h5py)."""

    def __init__(self,
                 tensorflow_session: tf.Session,
                 batch_size: int,
                 #keys_to_use: List[str],
                 npz_path: str,
                 testing=False,
                 eye_image_shape=(36, 60),
                 **kwargs):
        """Create queues and threads to read and preprocess data from specified keys."""
        mp2 = np.load(npz_path)
        # hdf5 = h5py.File(hdf_path, 'r')         # read h5 file
        self._short_name = 'NPZ:%s' % '/'.join(npz_path.split('/')[-2:])  # [-2:] only last 2 file route
        if testing:
            self._short_name += ':test'

        # Cache some parameters
        self._eye_image_shape = eye_image_shape

        # Create global index over all specified keys
        self._index_to_key = {}
        index_counter = 0
        n = mp2['gazeData'].shape[0]
        for i in range(n):
            self._index_to_key[index_counter] = i      # {index_counter, (train or test/p_id, i)}
            index_counter += 1
        # for key in keys_to_use:
        #     n = hdf5[key]['eye'].shape[0]
        #     for i in range(n):
        #         self._index_to_key[index_counter] = (key, i)       # {index_counter, (train or test/p_id, i)}
        #         index_counter += 1
        self._num_entries = index_counter

        self._npz = mp2
        self._mutex = Lock()                      # thread Lock()
        self._current_index = 0
        super().__init__(tensorflow_session, batch_size=batch_size, testing=testing, **kwargs)

        # Set index to 0 again as base class constructor called HDF5Source::entry_generator once to
        # get preprocessed sample.
        self._current_index = 0

    @property                      # decorator func
    def num_entries(self):         # overide
        """Number of entries in this data source."""
        return self._num_entries

    @property                      # decorator func
    def short_name(self):          # overide
        """Short name specifying source HDF5."""
        return self._short_name

    def cleanup(self):
        """Close HDF5 file before running base class cleanup routine."""
        super().cleanup()

    def reset(self):
        """Reset index."""
        with self._mutex:
            super().reset()
            self._current_index = 0

    def entry_generator(self, yield_just_one=False):
        """Read entry from HDF5."""
        try:
            while range(1) if yield_just_one else True:
                with self._mutex:
                    if self._current_index >= self.num_entries:
                        if self.testing:
                            break
                        else:
                            self._current_index = 0
                    current_index = self._current_index
                    self._current_index += 1

                index = self._index_to_key[current_index]
                data = self._npz
                entry = {}
                for name in ('leftEyeData', 'rightEyeData',  'gazeData'):
                    entry[name] = data[name][index, :, :, :]
                yield entry
        finally:
            # Execute any cleanup operations as necessary
            pass

    def preprocess_entry(self, entry):
        """Resize eye image and normalize intensities."""
        oh, ow = self._eye_image_shape
        left_eye = entry['leftEyeData']
        right_eye = entry['rightEyeData']
        gaze_point = entry['gazeData']


        #left_eye = cv.cvtColor(left_eye, cv.cv2.COLOR_BGR2GRAY)
        left_eye = cv.resize(left_eye, (ow, oh))
        left_eye = left_eye.astype(np.float32)
        left_eye *= 2.0 / 255.0
        left_eye -= 1.0
        right_eye = cv.resize(right_eye, (ow, oh))
        if (right_eye.ndim < 3):
            left_eye = np.expand_dims(left_eye, axis=0 if self.data_format == 'NCHW' else -1)  # add N dims to NCHW
        entry['leftEyeData'] = left_eye

        #right_eye = cv.cvtColor(right_eye, cv.cv2.COLOR_BGR2GRAY)
        right_eye = right_eye.astype(np.float32)
        right_eye *= 2.0 / 255.0
        right_eye -= 1.0
        if (right_eye.ndim < 3):
            right_eye = np.expand_dims(right_eye, axis=0 if self.data_format == 'NCHW' else -1)  # add N dims to NCHW
        entry['rightEyeData'] = right_eye


        entry['gazeData'] = gaze_point
        # entry['gazemaps'] = util.gazemap.from_gaze2d(
        #     entry['gaze'], output_size=(oh, ow), scale=0.5,
        # ).astype(np.float32)
        # if self.data_format == 'NHWC':
        #     np.transpose(entry['gazemaps'], (1, 2, 0))

        # Ensure all values in an entry are 4-byte floating point numbers
        for key, value in entry.items():
            entry[key] = value.astype(np.float32)

        return entry
