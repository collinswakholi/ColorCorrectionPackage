import os
import tempfile
import numpy as np
import pickle

import pytest

from ColorCorrectionPipeline.models import MyModels

def test_save_and_load_models(tmp_path):
    """
    Create a MyModels instance with dummy data, save it to disk,
    then load it into a new instance and compare attributes.
    """
    models = MyModels()
    # Assign dummy arrays / objects
    models.model_ffc = np.array([0.5, 1.0, 1.5])
    models.model_cc = {"dummy_cc": True}
    models.model_wb = np.eye(3)
    models.model_gc = [0.9, 2.2, 3.3]

    save_dir = tmp_path / "models_dir"
    save_dir.mkdir()
    models.save(str(save_dir), name="test_models")
    pkl_path = save_dir / "test_models.pkl"
    assert pkl_path.exists(), "Pickle file was not created."

    # Load into a fresh instance
    loaded = MyModels()
    loaded.load(str(save_dir), name="test_models")

    np.testing.assert_array_equal(loaded.model_ffc, models.model_ffc)
    assert loaded.model_cc == models.model_cc
    np.testing.assert_array_equal(loaded.model_wb, models.model_wb)
    assert loaded.model_gc == models.model_gc