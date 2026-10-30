"""Support the ColourChecker signatures used by supported Colour releases."""
import inspect

import colour


def make_colour_checker(name, data, illuminant, rows=None, columns=None):
    """Construct a chart without inventing geometry or masking library errors.

    Colour 0.4.1 exposes a three-field namedtuple; newer releases require rows
    and columns, which may be None when the source chart's layout is unknown.
    """
    constructor = colour.characterisation.ColourChecker
    parameters = inspect.signature(constructor).parameters
    options = dict(name=name, data=data, illuminant=illuminant)
    if "rows" in parameters:
        options["rows"] = rows
    if "columns" in parameters:
        options["columns"] = columns
    return constructor(**options)
