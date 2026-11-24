# Package structure and ownership

The public Python package is `ColorCorrectionPipeline`. Its implementation keeps
image interpretation, numerical recipes, application and artifact validation
at explicit boundaries. The repository is `ColorCorrectionPackage`.

## Maintained tree

```text
ColorCorrectionPipeline/
  __init__.py                 Public imports
  config.py                   New-fit selection, aliases and versioned config import
  config_groups.py            Immutable grouped options and one translation map
  pipeline.py                 Transactional fit, activation and application
  contracts.py                Owned images, references and observations
  colorimetry.py              Declared transfer, white-point and color operations
  observations.py             Fixed chart measurements
  fitting.py                  Fitting policies and fit-support checks
  stage_policies.py           Neutral-stage contracts, tone/WB models and apply_tone
  modern.py                   StageFit: the per-run stage fitters (FFC, GC, WB, CC)
  ui_fitting.py               UI-family solvers, features and UIPredictor
  ui_nn.py                    Ordered residual-network training arithmetic
  ui_native.py                Byte and high-precision image equations (leaf)
  ui_stages.py                FFC gain-surface fit and staged FFC replay
  ui_execution.py             Native image prediction: fused kernel, else reference
  detection/                  ONNX inference, chart geometry recipes, providers
  flat_field/                 White-plane detection and guarded gain estimation
  core/                       Regressor_Model with its LUT cache, CC application,
                              transforms and shared execution kernels
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
`checker_compat.py` and `conventional.py` modules protect supported library
signatures. `_compat/` holds the load-only code that keeps calibrations saved by
2.x and earlier loading and applying; nothing in it fits a new calibration.
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

## Module layers

Imports point downward. The numerical core has no import cycles, including
cycles hidden by function-local imports.

```text
pipeline --- acceleration              simple --> config --> config_groups,
   |   prediction, batch, results, reporting        fitting, contracts
   v
models --- bundle --> _compat
   |
modern.StageFit            the per-run stage fitters (FFC, GC, WB, CC)
   |
   +--> ui_execution       native image prediction (fused kernel, else reference)
   +--> observations ----> detection (charts, geometry, providers)
   +--> core.correction    Regressor_Model + LUT cache, CC tuples, predict_
   +--> core.execution     shared tone/LUT/affine kernels
   +--> ui_stages          FFC gain surface, staged FFC replay
           |
           v
        ui_fitting         fit_ui, UIPredictor --> ui_nn
           |
           v
leaves: ui_native, stage_policies (apply_tone), colorimetry, contracts, core.accel
```

Before the 3.0 structural cleanup, 23 modules formed one import cycle that
function-local imports hid. The stage fitters were spread over `modern`,
`ui_stages`, `ui_fitting` and `ui_native`, and they wrote private pipeline
fields (`_chart_index`, `_source_face`, `_ffc_domain`). `core/lut_cache.py`
wrote nine private `Regressor_Model` fields. `config` imported `simple`, and
detection imported the UI fitting modules for its geometry:

```text
before (<-> = mutual imports)               after
ui_native <-> ui_fitting <-> modern         modern.StageFit --> ui_stages --> ui_native
    ^            ^            ^             modern.StageFit --> core.correction --> ui_fitting
    v            v            v             ui_execution --> core.correction, ui_native
ui_stages <-> ui_execution <-> core.correction <-> core.lut_cache
prediction <-> bundle <-> models            (core.lut_cache merged into Regressor_Model)
config <-> simple                           simple --> config
detection --> ui_native, ui_fitting         detection --> detection.geometry
```

Two pairs still refer to each other through function-local imports:
`MyModels.save_bundle` and `load_bundle` delegate to `bundle`, which builds
`MyModels`; and `ColorCorrection.prepare_execution` builds an
`acceleration.PreparedExecution`, which hosts a fresh `ColorCorrection`.

## Boundaries that protect compatibility

The [grouped configuration API](grouped-configuration.md) is a presentation and
validation boundary over the existing flat `Config` state. Frozen `Steps` and
step option containers own their mapping inputs. Grouped constructor requests
normalize into the same stage dictionaries, policies and switches consumed by
the pipeline. Fresh read-only grouped snapshots reflect later supported flat
mutations; there is no mirrored mutable tree to keep synchronized.

`Config.to_dict()` keeps the existing flat contract used by runtime snapshots
and persistence. The explicit `grouped-v1` settings layout has its own marker
and API-2 interpretation. Unversioned flat mappings require
`source_version=2`; the API-1 profile was removed in 3.0.
Categorization does not change a feature order, numerical runtime or bundle
schema. Step-local ownership coexists with shared precision, color, reference,
observation and geometry checks. Export and prepared execution remain separate
public APIs rather than unused Config wrappers.

`FittingPolicy` owns the valid numerical profiles and runtime capabilities.
Its `sampling_policy` mapping is shared by configuration defaults and validation,
so adding a runtime cannot leave a separate sampling list out of sync.
New-fit default resolution is distinct from interpretation of an old settings
dictionary or a saved model. New fits use only the UI-compatible linear-RGB
recipe; saved historical encoded-RGB models apply with their recorded arithmetic
through `_compat/`.

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
- Import paths named by trusted historical model files remain resolvable, for
  example 1.x `key_functions` classes are remapped on unpickle. The historical
  fitting modules themselves were removed in 3.0 ([migration](migration-3.md)).
- Optional numerical kernels keep a reference path and capability checks.
  Backend presence alone does not establish numerical accuracy or GPU execution.
- README instructions, migration guidance and installed examples describe the
  same new-fit defaults. Saved historical recipes are labeled with their profile.
- Regression fixtures and generated qualification evidence have separate roles.
  The repository carries executable checks. Large generated outputs stay outside
  release archives.

See [distribution qualification](distribution-qualification.md) for installation
and release checks, [migration](migration-2.md) for public model/precision controls,
and [calibration bundles](calibration-bundles.md) for the persisted contract.
