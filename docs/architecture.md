# Package structure and ownership

The public Python package is `ColorCorrectionPipeline`. Its implementation keeps
image interpretation, numerical recipes, application and artifact validation
at explicit boundaries. The repository is `ColorCorrectionPackage`.

## Maintained tree

```text
ColorCorrectionPipeline/
  __init__.py                 Public imports
  config.py                   New-fit selection and versioned config import
  pipeline.py                 Transactional fit, activation and application
  contracts.py                Owned images, references and observations
  colorimetry.py              Declared transfer, white-point and color operations
  observations.py             Fixed chart measurements
  fitting.py                  Fitting policies and reproducible historical fits
  stage_policies.py           Neutral-stage contracts and numerical models
  modern.py                   Versioned stage orchestration
  ui_fitting.py               UI-family solvers, features and predictor validation
  ui_nn.py                    Ordered residual-network training arithmetic
  ui_native.py                Byte and high-precision image equations
  ui_stages.py                UI-family stage fitting
  ui_execution.py             Optional image kernels qualified against those equations
  detection/                  ONNX geometry, provider selection and execution evidence
  flat_field/                 White-plane detection and guarded gain estimation
  core/                       Historical numerical entry points and shared execution
  io/                         Strict decode, declared precision and verified export
  models.py                   Model container and activation-time validation
  bundle.py                   Bounded numerical serialization and content identity
  prediction.py, batch.py      Snapshot-based application and bounded batch work
  results.py, reporting.py     Owned outcomes and durable reports
  acceleration.py             Explicit prepared execution and accuracy budgets
  examples/                   Executable installed-package tutorials
docs/                         User contracts, migration and qualification
scripts/                      Independent distribution and installed-package checks
Images/                       Single source of the three packaged sample photographs
.github/workflows/            Distribution qualification and authorized publication
pyproject.toml                Build, dependency and package-resource declarations
```

Small numerical and worker helpers live beside their consumers. The retained
`checker_compat.py`, `conventional.py` and `key_functions.py` modules protect
supported library signatures and historical import/application contracts.
Trained source models, scratch results and interpreter/build caches are excluded
from the public archives. The two pinned ONNX inference assets carry their
provenance and notices.

## Calibration and application

A fresh `Config` selects the model, basis and numerical profile. File decoding
produces owned `ImageData` with its original precision, range, encoding and
orientation. The fitting profile resolves precision once from the chart and
enabled white field. Detection locates geometry in a preview, while the chosen
runtime measures samples from the declared image representation.

The pipeline fits an isolated candidate. Numerical reports record effective
settings, reference/observation identities, feature order and runtime. Only a
successful fit and requested persistence activate that candidate. An application
or batch works from a consistent model snapshot. A loaded calibration selects
its recorded numerical recipe.

The dependency direction is deliberate:

- Contracts and color conversion define the meaning of data.
- Numerical fitters consume those declared arrays and produce validated predictors.
- The pipeline coordinates decoding, detection, fitting and activation.
- Persistence validates numerical structures before constructing a candidate.
- Optional execution must agree with the reference predictor before it is selected.
- Examples consume public APIs; verification also uses independent fixtures and
  external numerical references.

## Boundaries that protect compatibility

`FittingPolicy` owns the valid numerical profiles and runtime capabilities.
Its `sampling_policy` mapping is shared by configuration defaults and validation,
so adding a runtime cannot leave a separate sampling list out of sync.
New-fit default resolution is distinct from interpretation of an old settings
dictionary or a saved model. Historical encoded-RGB and UI-native linear-RGB
training keep their recorded solvers and arithmetic.

`ui_fitting.py` owns feature generation for new linear, PLS and NN fits. The
predictor validates feature dimensions from that same versioned description.
Independent Finlayson comparisons in the qualifier check the mathematical
interpretation rather than repeating the feature builder.

`bundle.py` owns schema capabilities and bounded array decoding. Schema 1
preserves supported historical artifacts. Schema 2 identifies new configuration,
basis and precision capabilities. A content hash detects inconsistency, while
structural and semantic checks determine whether the calibration is supported.

The archive inspector deliberately keeps pinned ONNX hashes independent of the
runtime that loads those assets. Installed checks run outside the source tree.
That separation catches missing resources, accidental source imports and
packaging mistakes that an in-checkout test could miss.

## Maintenance decisions

- Resource access lives in the examples' shared helper. It uses the Traversable
  API on current Python and the supported legacy API only on Python 3.8.
- Historical modules and import paths remain explicit because old scripts and
  trusted historical model files can depend on them.
- Optional numerical kernels keep a reference path and capability checks.
  Backend presence alone does not establish numerical accuracy or GPU execution.
- README instructions, migration guidance and installed examples describe the
  same new-fit defaults. Historical recipes are labeled with their profile.
- Regression fixtures and generated qualification evidence have separate roles.
  The repository carries executable checks. Large generated outputs stay outside
  release archives.

See [distribution qualification](distribution-qualification.md) for installation
and release checks, [migration](migration-2.md) for public model/precision controls,
and [calibration bundles](calibration-bundles.md) for the persisted contract.

