# Pipeline contracts and color interpretation

The default `legacy-v1` path retains the earlier encoded-sRGB numerical behavior
and saved-model interpretation. GPU-first detection is now the default independently
of color semantics. Pin `chart_detection={"device":"cpu"}` when reproducing the
earlier CPU chart observations; inference arithmetic can slightly change geometry
and measurements across GPU providers.

```python
from ColorCorrectionPipeline import ColorCorrection, Config, ImageData, ReferenceProfile

# The typed boundary explicitly normalizes uint16 RGB and records its precision.
image = ImageData.from_array(rgb_uint16, color_space="srgb", channel_order="RGB")
config = Config(
    do_ffc=False,
    check_saturation=False,
    color_semantics={"version": "explicit-v1"},
    chart_detection={"device": "auto", "allow_cpu_fallback": True},
)
pipeline = ColorCorrection()
metrics, images, failed = pipeline.run(image, config=config)
```

## Ownership and observations

`ImageData`, `ReferenceProfile`, `ChartObservation` and `ObservationSet` own
immutable, byte-backed arrays. Deep copies and pickle/JSON restoration preserve
validation and immutability. Arrays cannot be made writable by toggling a flag.
Image metadata distinguishes source dtype/range/order/orientation from normalized
RGB working pixels; it never calls a decoded float64 image a 64-bit camera capture.

One run detects geometry from the original, normalized-orientation source.
Every stage samples the same canonical quadrilaterals and row-major `rNcM` cell
IDs. All detected Classic faces are retained in `pipeline.observations` and
`stage_observations`; `chart_index` selects the face used in ordinary and extended
fitting. IDs are stable within one capture/run, not inferred physical chart serial
numbers or guaranteed tracking IDs across independent detections.

In explicit mode, detection previews use uint8, while measured RGB is sampled from
full-precision working pixels. Legacy mode retains its earlier uint8/OpenCV
sampling. Observation means and clipped fractions use an 81-point grid per cell;
clipped fractions describe sampled pixels, not complete cell-region coverage.
The extended regression training grid has its separately configured `n_samples`.
An observation summary is not the full training dataset or an independent label
for every repeated pixel. Missing geometry is recorded as unavailable, never zero
measurements or fabricated cells. MCC remains a legacy route without model quads.

## Canonical references and explicit color semantics

`ReferenceProfile` stores unclipped canonical Lab D50, human labels, canonical cell
IDs, declared neutrals, layout/template, source interpretation, revision and a
content digest. Known templates require canonical row-major IDs. Custom values
must declare their RGB/Lab/XYZ encoding, white and scale; no row-order or scale
guessing occurs. Custom profiles require explicit pipeline semantics and Classic24
layout today. Display RGB is derived separately and can clip; it never replaces
reference truth or the unclipped fitting targets.

The default catalog profile is derived from colour-science's after-November-2014
Classic24 xyY table, with its library version/provenance retained. It is not
described as an independently verified manufacturer CGATS file.

`explicit-v1` uses sRGB primaries, encoded sRGB/D65 inside the correction engine,
canonical Lab D50, declared Bradford (default) or CAT02 adaptation, and a chosen
common D50/D65 evaluation white. RGB transfer, matrix, CAT and Lab operations are
explicit. Gamma retains the same all-patch polynomial fitting/selection algorithm,
but explicit mode uses canonical working-white lightness and pure D65 Lab fit/apply
conversion. This avoids an out-of-gamut RGB round-trip and dependence on the global
colour-science domain/range setting. Legacy gamma keeps its earlier fast conversion.

Declared neutral IDs drive explicit WB; legacy WB defaults to the last six cells.
The historical 0.95 headroom factor remains. CC still fits encoded RGB; selecting
linear input/output does not silently change its regression features or domain.
The current pipeline requires output policy `clip` because its stages clip. Other
policies are available in pure conversion helpers but rejected by this engine.

Explicit metrics compare actual fixed-region output samples against canonical Lab
truth at one declared evaluation white using CIEDE2000. They report training-chart
residuals, not independent calibration accuracy. Legacy metrics retain their old
behavior, including the historical `REF_ILLUMINANT` interpretation.

## Encoding, file decoding and persistence

A typed image carries its own encoding and is converted to working encoded sRGB.
Raw arrays use configured `input_encoding`; raw prediction arrays must use the
saved model's configured encoding. Unwrapping typed linear pixels into an array
does not retain their type metadata automatically. `output_encoding='linear-srgb'`
converts returned stage images at the boundary. These settings persist and govern
new pipeline instances that load the model.

For explicit raw white inputs, file/uint8/float forms consistently use the configured
input encoding. The historical uint8-white BGR channel order is retained; typed
white data declares its own order/encoding. FFC estimation still converts the white
image to uint8 BGR and records that actual precision. Higher-precision radiometric
FFC is later work.
Explicit application also requires the calibrated FFC image dimensions; it does
not silently resize a spatial gain map. This shape gate does not certify matching
lens, crop, illumination or capture conditions. Legacy application retains its
historical resizing behavior.

`read_image_data` reads uint8/uint16 three-channel images with unchanged precision
and applies EXIF orientation once. Pending nonidentity orientation must be explicitly
normalized with `with_orientation_applied()` before processing. Embedded ICC profiles
are rejected until the caller converts them or explicitly chooses
`icc_policy='assume-declared'`. That decision/profile hash is recorded; no implicit
ICC transformation is claimed. Grayscale, alpha and multiframe inputs require a
caller-selected conversion before import. Legacy file decoding remains unchanged.

Legacy pickle containers are still trusted-file-only. New metadata stores image,
reference, observation and color contracts, stage switches, detector provenance
and algorithm identities. Loading validates hashes/assignments and supported
engine policies before changing active models. Metadata-free older models retain
legacy interpretation, without invented source/capture metadata. Use the separate [portable bundle format](calibration-bundles.md) for
non-executable inference parameters. Old package versions do not understand new
explicit metadata.

The defaults described here retain the earlier stage algorithms. Opt-in modern
saturation/neutral GC/WB/guarded FFC and reproducible fitting are now documented
in [numerical policies](numerical-policies-and-validation.md).
Independent-capture validation and high-precision FFC remain later work.
Fit/apply parity comparisons must account for the legacy fit-only saturation operation.
