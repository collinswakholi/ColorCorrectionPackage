"""
MyModels: A simple container for storing and loading trained models:
- Flat-field multiplier
- Color‐correction (matrix or model)
- White‐balance diagonal matrix
- Gamma correction coefficients
"""

import os
import pickle
from typing import Any, Optional


class MyModels:
    def __init__(self) -> None:
        self.model_ffc: Optional[Any] = None  # Flat-Field Correction multiplier
        self.model_cc: Optional[Any] = None   # Color correction model/matrix
        self.model_wb: Optional[Any] = None   # White-balance diagonal matrix
        self.model_gc: Optional[Any] = None   # Gamma correction coefficients

    def save(self, directory: str, name: str = "models") -> None:
        """
        Save all models to a pickle file in `directory/name.pkl`.
        """
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{name}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    def load(self, directory: str, name: str = "models") -> None:
        """
        Load models from a pickle file in `directory/name.pkl` into self.
        """
        filepath = os.path.join(directory, f"{name}.pkl")
        with open(filepath, "rb") as f:
            loaded: MyModels = pickle.load(f)

        self.model_ffc = loaded.model_ffc
        self.model_cc = loaded.model_cc
        self.model_wb = loaded.model_wb
        self.model_gc = loaded.model_gc
