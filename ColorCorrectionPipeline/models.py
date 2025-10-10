"""
Model storage and persistence module
=====================================

Provides a container for storing and loading trained correction models.
Models include flat-field correction multipliers, gamma correction coefficients,
white balance matrices, and color correction models/matrices.
"""

import os
import pickle
from pathlib import Path
from typing import Any, Optional

__all__ = ["MyModels"]


class MyModels:
    """
    Container for storing and persisting color correction models.

    This class stores trained models from each stage of the pipeline:
    - Flat-field correction (FFC) multiplier
    - Gamma correction (GC) coefficients
    - White balance (WB) diagonal matrix
    - Color correction (CC) model or matrix

    Attributes:
        model_ffc: Flat-field correction multiplier (numpy array).
        model_gc: Gamma correction coefficients (numpy array).
        model_wb: White balance diagonal matrix (numpy array).
        model_cc: Color correction model (matrix or trained model object).

    Example:
        >>> models = MyModels()
        >>> # After training...
        >>> models.model_ffc = ffc_multiplier
        >>> models.model_gc = gc_coeffs
        >>> models.model_wb = wb_matrix
        >>> models.model_cc = (ccm, params, "ours")
        >>> 
        >>> # Save models
        >>> models.save("results", "my_models")
        >>> 
        >>> # Load models later
        >>> loaded_models = MyModels()
        >>> loaded_models.load("results", "my_models")
    """

    def __init__(self) -> None:
        """Initialize empty model container."""
        self.model_ffc: Optional[Any] = None
        self.model_gc: Optional[Any] = None
        self.model_wb: Optional[Any] = None
        self.model_cc: Optional[Any] = None

    def save(self, directory: str, name: str = "models") -> None:
        """
        Save all models to a pickle file.

        Args:
            directory: Directory path where models will be saved.
            name: Base filename for the pickle file (without extension).

        Raises:
            OSError: If directory cannot be created or file cannot be written.

        Example:
            >>> models.save("results", "experiment_01")
            # Creates: results/experiment_01.pkl
        """
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)
        
        filepath = directory_path / f"{name}.pkl"
        
        try:
            with open(filepath, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise OSError(f"Failed to save models to {filepath}: {e}") from e

    def load(self, directory: str, name: str = "models") -> None:
        """
        Load models from a pickle file.

        This method loads a previously saved MyModels instance and copies
        all model attributes to the current instance.

        Args:
            directory: Directory path where models are stored.
            name: Base filename of the pickle file (without extension).

        Raises:
            FileNotFoundError: If the pickle file does not exist.
            OSError: If file cannot be read.
            pickle.UnpicklingError: If pickle file is corrupted.

        Example:
            >>> models = MyModels()
            >>> models.load("results", "experiment_01")
            # Loads from: results/experiment_01.pkl
        """
        filepath = Path(directory) / f"{name}.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            with open(filepath, "rb") as f:
                loaded: "MyModels" = pickle.load(f)
        except Exception as e:
            raise OSError(f"Failed to load models from {filepath}: {e}") from e

        # Copy attributes from loaded instance
        self.model_ffc = loaded.model_ffc
        self.model_gc = loaded.model_gc
        self.model_wb = loaded.model_wb
        self.model_cc = loaded.model_cc

    def has_models(self) -> bool:
        """
        Check if any models have been trained/loaded.

        Returns:
            True if at least one model is not None, False otherwise.
        """
        return any(
            [
                self.model_ffc is not None,
                self.model_gc is not None,
                self.model_wb is not None,
                self.model_cc is not None,
            ]
        )

    def clear(self) -> None:
        """Clear all stored models."""
        self.model_ffc = None
        self.model_gc = None
        self.model_wb = None
        self.model_cc = None

    def __repr__(self) -> str:
        """String representation showing which models are loaded."""
        models_status = {
            "FFC": self.model_ffc is not None,
            "GC": self.model_gc is not None,
            "WB": self.model_wb is not None,
            "CC": self.model_cc is not None,
        }
        loaded = [name for name, status in models_status.items() if status]
        return f"MyModels(loaded=[{', '.join(loaded)}])"
