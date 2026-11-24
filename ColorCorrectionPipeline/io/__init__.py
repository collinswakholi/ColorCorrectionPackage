"""
Image input/output with explicit encoding, range and precision contracts
========================================================================

- read_image_data: strict decoding into ImageData (the pipeline's reader)
- write_image_data: strict, verified export returning an ExportRecord

3.0 removed the 1.x-style read_image, read_batch and write_image helpers; see
docs/migration-3.md.
"""

from .writers import write_image_data, ExportRecord
from .image_data import read_image_data

__all__ = [
    "read_image_data",
    "write_image_data",
    "ExportRecord",
]
