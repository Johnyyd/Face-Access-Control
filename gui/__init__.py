"""
Face Access Control - GUI Package
Export GUI classes
"""

from .main_window_gradio import GradioMainWindow
from .main_window_tkinter import TKinterMainWindow

__all__ = ["GradioMainWindow", "TKinterMainWindow"]
