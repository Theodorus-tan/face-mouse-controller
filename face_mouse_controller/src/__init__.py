"""
Face-Controlled Mouse System
A computer vision-based mouse control system using facial gestures.
"""

__version__ = "1.0.0"
__author__ = "Song Tan (Ningbo University)"

from .config import Config
from .filters import KalmanFilter, MovingAverageFilter
from .face_detector import FaceDetector
from .mouse_controller import MouseController
from .application import FaceMouseApplication

__all__ = [
    "Config",
    "KalmanFilter",
    "MovingAverageFilter",
    "FaceDetector",
    "MouseController",
    "FaceMouseApplication",
]
