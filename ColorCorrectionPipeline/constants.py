"""
Constants module for ColorCorrectionPipeline
============================================

This module contains package-wide constants to avoid circular imports.
"""

import os

# Define the absolute path to the YOLO model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "FFC", "Models", "plane_det_model_YOLO_512_n.pt")

__all__ = ['MODEL_PATH']