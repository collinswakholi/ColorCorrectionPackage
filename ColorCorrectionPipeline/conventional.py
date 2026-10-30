"""Stored-matrix application across supported colour-science API generations.

Current versions delegate unchanged. Older versions use the same published
feature expansion and final dot expression, without refitting a saved matrix.
"""
import numpy as np


def conventional_basis(RGB, method="Finlayson 2015", **options):
    import colour
    library = colour.characterisation
    def expand_with(function, **keywords):
        values = np.asarray(RGB)
        # Colour 0.4.1 squeezes singleton polynomial terms inconsistently.
        # Duplicate only for that legacy API and retain the original row.
        if not callable(getattr(library, "apply_matrix_colour_correction", None)) and values.shape == (1, 3):
            return function(np.repeat(values, 2, axis=0), **keywords)[:1]
        return function(RGB, **keywords)
    methods = ("Cheung 2004", "Finlayson 2015", "Vandermonde")
    if not isinstance(method, str) or method.lower() not in tuple(value.lower() for value in methods):
        raise ValueError("unsupported conventional colour correction method")
    key = method.lower()
    if key == "cheung 2004":
        expand = getattr(library, "polynomial_expansion_Cheung2004", None)
        if expand is None:
            expand = library.matrix_augmented_Cheung2004
        return expand_with(expand, **{name: options[name] for name in ("terms",) if name in options})
    if key == "finlayson 2015":
        return expand_with(library.polynomial_expansion_Finlayson2015,
            **{name: options[name] for name in ("degree", "root_polynomial_expansion") if name in options})
    return expand_with(library.polynomial_expansion_Vandermonde,
        **{name: options[name] for name in ("degree",) if name in options})


_MISSING = object()


def apply_conventional(RGB, CCM, method=_MISSING, **options):
    import colour
    apply = getattr(colour.characterisation, "apply_matrix_colour_correction", None)
    if callable(apply):
        return apply(RGB, CCM, **options) if method is _MISSING else apply(RGB, CCM, method=method, **options)
    # This is the exact stored-matrix application expression in the historical
    # colour_correction_* functions, excluding their matrix refitting step.
    from colour.utilities import as_float_array
    values = as_float_array(RGB)
    shape = values.shape
    expanded = conventional_basis(values.reshape(-1, 3), method="Cheung 2004" if method is _MISSING else method, **options)
    return np.reshape(np.transpose(np.dot(CCM, np.transpose(expanded))), shape)
