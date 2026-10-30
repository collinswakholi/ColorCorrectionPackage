# Image, reference and observation contracts

`ColorCorrectionPipeline.contracts` provides version-1 objects describing owned
pixels, canonical reference truth, and detected chart observations. These objects
make interpretation and identity inspectable without replacing the legacy
ndarray APIs or implying a different regression algorithm.

## ImageData

```python
from ColorCorrectionPipeline.contracts import ImageData

image = ImageData.from_array(
    bgr_uint16,
    channel_order="BGR",
    color_space="srgb",
    orientation=1,
    provenance={"reader": "explicit uint16 input"},
)
working_rgb = image.to_srgb()
metadata = image.to_metadata()  # no full pixel list
```

Canonical `pixels` are RGB float64 in `[0,1]`, interpreted as declared encoded
`srgb` or `linear-srgb`. The contract does not infer encoding from brightness,
perform ICC conversion, or silently clip source values. `from_array` supports
uint8, uint16, float16/32/64; signed integers, booleans, complex values, empty or
non-RGB shapes, nonfinite values and ambiguous encodings fail explicitly.

Integer defaults use their full dtype range: 255 or 65535. Floating defaults
mean `[0,1]`; a floating 0–255 image requires `value_range=(0,255)`. An explicitly
declared alternate range, such as a 12-bit range in a uint16 container, can be
provided. Precision is retained in canonical float storage until a caller
deliberately requests an 8-bit view.

All eight EXIF orientations can be applied once by `from_array`. Source dtype,
range, channel order, source shape, orientation code and applied status are
recorded. `apply_orientation=False` records an unapplied orientation rather than
pretending the image has been normalized. A file loader must know whether its
decoder already applied orientation; do not apply both decoder orientation and
this adapter's orientation to the same pixels.
`with_orientation_applied()` explicitly normalizes a pending orientation once
while preserving the original dtype/range/shape facts; repeated calls are no-ops.

`to_srgb()` returns an independent writable RGB array suitable for current
correction stages. It encodes linear-sRGB inputs explicitly. `to_bgr8()` creates
an explicit nearest-integer detector view; it does not overwrite high-precision
canonical storage. Path loading, ICC policy and export capabilities are separate
I/O concerns, not inferred by this contract.

`pixel_digest` identifies normalized pixels and their encoding. `digest` also
records source interpretation/provenance. Thus uint8 and floating inputs with
identical canonical pixels can share pixel identity while retaining different
source provenance. `to_metadata()` is small. `to_dict()` deliberately includes a
full pixel payload for explicit round-trips and should not be used merely to log
a large photograph.

## ReferenceProfile

`ReferenceProfile.classic24()` returns the maintained after-November-2014
Classic24 catalog, with stable row-major `r1c1` through `r4c6` cell IDs and named
human-readable labels. The six neutrals are `r4c1` through `r4c6`;
`neutral_indices` identifies their positions. Human labels may repeat; cell IDs
must be unique. A template ID and `(rows,columns)` layout prevent another 24-cell
chart from being accepted just because its cell count matches.
Known Classic, Spyder and Digital SG template IDs require canonical `rNcM`
row-major cell IDs in both reference profiles and observations. Reversing labels
cannot relabel fixed detector geometry. Generic custom templates may use their
own explicitly declared IDs. Custom imported values/names are supplied in
canonical row-major order; column-major import conversion is not implicit.

Reference truth is stored as **unclipped Lab D50**. Display swatches are derived
by `display_rgb(clip=True)` and never replace canonical values. The default
catalog uses fixed xyY source facts converted without colour-science's global
domain/range setting, so another caller changing that global setting cannot
change the canonical profile or its digest.

Use `from_values` to resolve custom values once with explicit source semantics:

```python
reference = ReferenceProfile.from_values(
    values,
    color_space="lab", white="D50", scale=1,
    adaptation="Bradford",
    profile_id="my-classic-reference", revision=1,
    names=names, cell_ids=cell_ids, neutral_ids=neutral_ids,
    layout=(4,6), template_id="classic_4x6",
)
```

The colorimetry layer validates supported Lab, XYZ, encoded-sRGB and linear-sRGB
source semantics. Source interpretation, adaptation choice, edition/provenance,
profile ID/revision and canonical truth enter the profile digest. Canonical
arrays are finite but are not display-gamut clipped. Recording nominal values
does not certify a photographed chart's age, cleanliness, illumination or
physical calibration.

## ChartObservation and ObservationSet

Each `ChartObservation` records one detected face: chart/face IDs, template,
ordered cell IDs, quadrilaterals, measured RGB, source image dimensions,
confidence, stage name, reference digest, clipping quality and provenance.
Geometry must contain distinct nondegenerate convex quadrilaterals inside the
declared image. Structural validation does not independently prove that an
operator assigned the right physical chart or cell names.

`ChartObservation.from_samples(samples=...)` accepts `(cells,samples_per_cell,3)`
normalized RGB. It computes patch means and the fraction of sampled pixels with
any channel at/above the declared clipping threshold. A partially clipped patch
is therefore marked even if its mean is below the threshold. The recorded basis
is sampled pixels, not full-region clipping coverage. Cell status can explicitly
be measured, clipped, excluded or estimated; estimated values must not be
presented as independent measurements.

`validate_reference(profile)` checks the exact profile digest, template, layout,
and ordered cell IDs. Reversing IDs or substituting a different profile revision
cannot silently keep the assignment valid. Multiple face observations remain
separate in an `ObservationSet`; they are not averaged or relabeled implicitly.

`ObservationSet.select_face()` chooses highest confidence, using area and stable
IDs for deterministic ties. `select_face("largest")` is explicit. Specific
chart/face selection rejects ambiguous identity matches. An empty set requires
an `unavailable_reason`, allowing disabled fitting or legacy extraction without
inventing unaudited quadrilaterals or zero-valued chart measurements.

## Ownership and interchange

All arrays have immutable bytes backing. Caller mutation cannot alter a
contract, and `array.setflags(write=True)` fails. Nested provenance is recursively
owned and immutable. Copy/deepcopy reuse these immutable snapshots. Pickle
round-trips reconstruct and revalidate the immutable backing, rather than
accepting NumPy's ordinary writable restoration.

`to_dict`/`from_dict` round-trip through JSON-compatible values. Unknown fields,
unknown/noninteger versions, nonfinite values, inconsistent shapes and supplied
digest mismatches are rejected. A digest is computed when importing a new
payload without one; that does not establish external authenticity. These data
contracts do not make arbitrary pickle files safe to load. Portable inference
parameters use the separate [calibration bundle](calibration-bundles.md) format.
