"""Data-source definitions (one class per file)."""
from .frames import FramesSource
from .hdf5 import HDF5Source
from.hdf5_iris import HDF5IrisSource
from .unityeyes import UnityEyes
from .video import Video
from .webcam import Webcam
from .image import Image
from .unityeyes_gazemap import UnityEyes_GazeMap
from .mp2_npz import NPZSource
__all__ = ('FramesSource', 'HDF5Source', 'UnityEyes', 'Video', 'Webcam', 'Image', 'UnityEyes_GazeMap', 'HDF5IrisSource', 'NPZSource')
