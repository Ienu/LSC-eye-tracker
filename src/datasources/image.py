"""single image for gaze estimation."""
import cv2 as cv

from .frames import FramesSource


class Image(FramesSource):
    """Read image from file and preprocessing."""

    def __init__(self, file_name, **kwargs):
        """Create queues and threads to read and preprocess data."""
        self._short_name = 'Webcam'

        self._file_name = file_name

        # Call parent class constructor
        super().__init__(**kwargs)

    def frame_generator(self):
        """Read frame from file."""
        while True:
            bgr = cv.imread(self._file_name)
            #cv.waitKey(1)
            yield bgr
