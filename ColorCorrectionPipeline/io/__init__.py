"""
I/O module for color_correc_optim
==================================

This module provides input/output utilities for reading and writing images
with proper color space handling.

Submodules:
    - readers: Image reading utilities
    - writers: Image writing utilities
"""

from .readers import read_batch, read_image
from .writers import write_image

__all__ = ["read_image", "write_image", "read_batch"]
