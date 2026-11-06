# Pipeline contracts and color interpretation

New calibrations default to UI-native RPCC2 with linear-light fitting and
calibration-time precision selection. Explicit `legacy-v1` and
`reproducible-v1` profiles preserve their encoded-sRGB training recipes.
Saved calibrations restore their recorded interpretation. The
[migration guide](migration-2.md) describes those choices.

GPU-first detection is independent of the fitting domain. Pin
`chart_detection={"device":"cpu"}` when reproducing earlier CPU chart
observations. Inference arithmetic can slightly change geometry and measurements
across GPU providers.

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

The explicit historical reference catalog is derived from colour-science's
after-November-2014 Classic24 xyY table, with its library version/provenance
retained. The UI-native profiles use the retained UI catalog coordinates.
Custom physical XYZ/Lab references retain their declared coordinates when
requested; see [UI-native reference handling](ui-compatible-fitting.md).

`explicit-v1` declares sRGB primaries, encoded sRGB/D65 working images, canonical
Lab D50 references, and a common evaluation white. The fitting policy determines
the feature domain and the qualified adaptation. UI-native fitting uses Bradford
and linear RGB. Historical explicit gamma uses its all-patch polynomial recipe
with canonical working-white lightness and D65 Lab conversion. Modern profiles
use the neutral-stage algorithms recorded by their stage policy.

Declared neutral IDs drive explicit WB; historical WB defaults to the last six
cells and preserves its headroom convention. The legacy and reproducible CC
profiles fit encoded RGB, while UI-native CC fits linear RGB. Input/output
encoding is a separate declaration. The stage engine requires output policy
`clip`; pure conversion helpers also expose preserving policies.

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
white data declares its own order/encoding. Historical explicit policies estimate
FFC through their recorded uint8/float32 paths. The 2.0 high-precision runtime
measures the original white values in float64; see the [migration guide](migration-2.md).
This remains a Lab-lightness correction with the declared capture assumptions.
Explicit application also requires the calibrated FFC image dimensions; it does
not silently resize a spatial gain map. This shape gate does not certify matching
lens, crop, illumination or capture conditions. Legacy application retains its
historical resizing behavior.

`read_image_data` preserves supported uint8/uint16 samples and floating TIFF
values and applies EXIF orientation once. Pending nonidentity orientation must
be normalized with `with_orientation_applied()` before pipeline processing.
Embedded ICC profiles require a declared policy. The strict loader can convert
qualified uint8 profiles with `icc_policy='convert-srgb'`, or record an explicit
`'assume-declared'` decision. Unsupported high-precision profile conversions
fail before a byte downgrade. Alpha, grayscale and multiframe inputs require
their explicit loader policy or frame selection. See [strict image I/O](image-io-policies.md).

Legacy pickle containers are still trusted-file-only. New metadata stores image,
reference, observation and color contracts, stage switches, detector provenance
and algorithm identities. Loading validates hashes/assignments and supported
engine policies before changing active models. Metadata-free older models retain
legacy interpretation, without invented source/capture metadata. Use the separate [portable bundle format](calibration-bundles.md) for
non-executable inference parameters. Old package versions do not understand new
explicit metadata.

The [numerical policy guide](numerical-policies-and-validation.md) describes
reproducible encoded-RGB fitting and grouped evaluation. The
[UI-native guide](ui-compatible-fitting.md) specifies the default neutral stages,
joint FFC/GC rendering and linear-light CC.
Independent-capture validation requires a separately measured capture. The 2.0
high-precision FFC path has separate tests for original white-field values and
is identified in the stored runtime.
Fit/apply parity comparisons must account for the legacy fit-only saturation operation.
