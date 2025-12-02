"""
Face Access Control - Modules Package
Export các class chính để dễ dàng import
"""

from .camera import CameraManager
from .detector import FaceDetector
from .recognizer_lbph import LBPHRecognizer
from .recognizer_openface import OpenFaceRecognizer
from .recognizer_sface import SFaceRecognizer
from .detector_yunet import YuNetDetector
from .database import Database

__all__ = [
    'CameraManager',
    'FaceDetector',
    'LBPHRecognizer',
    'OpenFaceRecognizer',
    'SFaceRecognizer',
    'YuNetDetector',
    'Database'
]

__version__ = '1.0.0'
__author__ = 'Face Access Control Team'
