"""Preserve old/current ColourChecker APIs without catching real errors."""
from collections import namedtuple

import colour
import numpy as np
import pytest

from ColorCorrectionPipeline.checker_compat import make_colour_checker


def test_three_field_colour_checker_ignores_unsupported_grid_keywords(monkeypatch):
    constructor = namedtuple("ColourChecker", ("name", "data", "illuminant"))
    monkeypatch.setattr(colour.characterisation, "ColourChecker", constructor)
    data, illuminant = {"cell": np.array([.2, .3, .4])}, np.array([.3127, .3290])
    checker = make_colour_checker("reference", data, illuminant, rows=4, columns=6)
    assert checker.name == "reference" and checker.data is data
    assert checker.illuminant is illuminant and not hasattr(checker, "rows")


@pytest.mark.parametrize("rows,columns", [(4, 6), (None, None)])
def test_current_constructor_preserves_explicit_or_unknown_grid(monkeypatch, rows, columns):
    constructor = namedtuple("ColourChecker", ("name", "data", "illuminant", "rows", "columns"))
    monkeypatch.setattr(colour.characterisation, "ColourChecker", constructor)
    checker = make_colour_checker("reference", {}, np.array([.3127, .3290]), rows, columns)
    assert checker.rows == rows and checker.columns == columns


def test_constructor_internal_type_error_is_not_hidden(monkeypatch):
    def constructor(name, data, illuminant):
        raise TypeError("constructor internal failure")
    monkeypatch.setattr(colour.characterisation, "ColourChecker", constructor)
    with pytest.raises(TypeError, match="constructor internal failure"):
        make_colour_checker("reference", {}, np.array([.3127, .3290]))
