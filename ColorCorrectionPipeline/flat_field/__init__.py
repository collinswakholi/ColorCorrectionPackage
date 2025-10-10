"""
Flat-field correction module
=============================

This module provides flat-field correction utilities for removing illumination
non-uniformities from images.

Main class:
    - FlatFieldCorrection: Automatic and manual flat-field correction
"""

from .correction import FlatFieldCorrection

__all__ = ["FlatFieldCorrection"]
