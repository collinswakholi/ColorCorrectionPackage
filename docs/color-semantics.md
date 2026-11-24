# Explicit color semantics and legacy compatibility

New fits use `ColorSemantics(version="explicit-v1")`, the declared
conversion/reference/evaluation contract, with Bradford adaptation and
`fitting_domain="linear-rgb"`; `Config` selects it. `ColorSemantics()` still
identifies `legacy-v1`, the CAT02 encoded-sRGB interpretation recorded by saved
historical models. Those models load and apply unchanged, but `legacy-v1` can
no longer be fitted (removed in 3.0; see [Migrating to 3.0](migration-3.md)).

Input/output encoding support is separate from the fitting domain. The pipeline
must reject an output policy its stages cannot honor.

## Declared quantities

- RGB has sRGB primaries and either sRGB's piecewise transfer or an explicitly
  linear encoding. Only those encodings are implemented in this contract.
- Working white D65 is the package convention xy=(0.31271,0.32902). Canonical
  reference white D50 is xy=(0.3457,0.3585). Evaluation white is explicitly D50
  or D65 and is shared by measured and reference colors.
- XYZ is relative with white Y=1, or explicitly scale100. Lab always has L*
  units0..100 and scale1. Reference RGB accepts explicit scales1/255/65535.
- Adaptation is explicitly Bradford or CAT02. Their matrix inverse uses full
  floating precision. Converting XYZ to Lab under another white is not, by
  itself, chromatic adaptation.
- Canonical Lab D50 is reference truth. Encoded fitting targets may be outside
  `[0,1]`. Display swatches are a separate clipped view and never overwrite truth.

The matrix derives from sRGB primaries at the declared package D65. W3C's
normative RGB D65 commonly uses xy=(0.3127,0.3290); the small difference is
documented rather than hidden. Published matrix tests allow2e-4 absolute
coefficient tolerance for white rounding; internal round trips use much tighter
limits. This does not claim exact ICC conformance for the package's rounded white.

## Public conversion helpers

`ColorCorrectionPipeline.colorimetry` provides:

```python
semantics = ColorSemantics(version="explicit-v1", adaptation="Bradford")
targets = semantics.reference_to_working_rgb(canonical_lab_d50, clip=False)
target_lightness = semantics.reference_to_working_lab(canonical_lab_d50)[:, 0]
display = display_reference_rgb(canonical_lab_d50, adaptation="Bradford")
scores = semantics.delta_e00(measured_rgb, canonical_lab_d50)
```

`reference_values_to_lab_d50(values, color_space=..., white=..., scale=...,
adaptation=...)` canonicalizes Lab/XYZ/sRGB/linear-sRGB reference inputs. Supplied
Lab D50 values are copied exactly instead of being transformed through display
RGB. RGB reference encodings use the declared D65; other whites must be
represented as Lab/XYZ rather than relabeling standard RGB.

Lightness targets use a direct canonical-Lab→XYZ→CAT→working-Lab path. They must
not be reconstructed by decoding unclipped reference RGB with a legacy transfer
function: negative-channel extensions can differ. For canonical LabD50
`[65,120,-110]`, direct working L* is66.13847, while the legacy RGB round trip
produces68.82213. This bypass preserves canonical meaning.

Pure `rgb_to_xyz`, `xyz_to_rgb`, `xyz_to_lab`, `lab_to_xyz`, and `adapt_xyz`
helpers name their units/whites. `xyz_to_rgb` expects XYZ already at D65; it does
not perform an implicit CAT. `srgb_to_linear` defaults to the unit input domain.
An explicit extended domain enables a signed transfer extension. Encoding output
policies are `clip`, `preserve`, or `raise`; display clipping is never implicit
in reference canonicalization.

`delta_e00_reference` decodes the declared measured encoding, adapts measured
D65 XYZ and canonical D50 XYZ to one evaluation white, then computes CIEDE2000.
It requires same-shape arrays representing the same cells. Interpretation
cannot be inferred from channel magnitudes. Treating linear pixels as encoded
sRGB produces a demonstrably different score.

The legacy `core.color_spaces` helpers (`convert_to_lab`, `adapt_chart`,
`do_color_adaptation` and related functions) were removed in 3.0. Use the
`colorimetry` functions above for encoding/white conversion.

## Verification and sources

The transfer functions, primary matrices, and CIE Lab invariants follow the
[W3C color-conversion reference](https://www.w3.org/TR/css-color-4/#color-conversion-code).
Tests cover published vectors, transfer breakpoints, black/white, scale agreement,
RGB/XYZ and CAT round trips, finite/unit validation, same-color scoring in both
encodings, and a deliberate double-adaptation counterexample.

The CIEDE2000 implementation is tested in both directions against all34 published
Sharma/Wu/Dalal supplementary pairs, to0.00005 (the source's four-decimal
precision). The fixture is preserved from the comparison UI's independently
attributed test data. [Author's paper and test dataset](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/).
This validates the numerical formula and units, not a physical chart or camera.

No independent physical color-accuracy improvement is claimed by these software
checks. A later full-pipeline validation and capture protocol is still required.

The [UI-compatible fitting policy](ui-compatible-fitting.md) explicitly
selects `fitting_domain="linear-rgb"`. An input encoding alone does not select
that algorithm.
