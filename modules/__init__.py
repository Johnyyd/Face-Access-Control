"""
Face Access Control - Modules Package
Export các class chính để dễ dàng import
"""

from .camera import CameraManager
from .detector import FaceDetector
from .database import Database
from .recognizer_lbph import LBPHRecognizer
from .recognizer_openface import OpenFaceRecognizer

__all__ = [
    'CameraManager',
    'FaceDetector',
    'Database',
    'LBPHRecognizer',
    'OpenFaceRecognizer'
]

__version__ = '1.0.0'
__author__ = 'Face Access Control Team'
