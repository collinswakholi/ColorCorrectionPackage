# Non-executable calibration bundle version 1

The `.ccpb` format stores declarative inference parameters and recorded calibration
metadata. It does not store Python objects, executable graphs, optimizer/training
state or runtime callbacks. Existing `MyModels.save/load` pickle APIs remain the
trusted-file compatibility route. The new bundle loader never falls back to pickle.

```python
pipeline.models.save_bundle("calibration.ccpb")
restored = MyModels()
restored.load_bundle("calibration.ccpb")
```

Loading constructs and validates a fresh candidate before the caller activates it.
Saving completes serialization, flush/fsync and a validated temporary-file readback
beside the destination, then atomically replaces that destination. A failure leaves
the existing file and active model set intact.

## What is preserved

- FFC spatial Lab-lightness gain arrays, polynomial GC or neutral PCHIP knots,
  WB matrices, and conventional CC matrices with a named supported feature basis.
- Polynomial regressor degree/feature order, direct inference parameters, LUT
  settings and any already-built LUT values.
- Linear coefficients/intercept; explicit PLS centering/scaling equation; dense
  sklearn MLP weights, biases and known activation functions; package Torch NN
  weights/biases and eval-time Linear/BatchNorm/activation/dropout semantics.
- UI-compatible linear-light feature order, ridge/PLS affine parameters and
  residual NN arrays, with matching versioned fitting/color metadata.
- Calibration policies, reference/observation contracts, capture assumptions,
  algorithm versions, detector identities, fitting/evaluation records and existing
  dependency provenance. Export runtime versions are recorded separately.

Portable Torch inference uses fixed CPU float32 functional operations, including
BatchNorm running statistics, without module initialization or RNG use. It does not
preserve a training device or optimizer. Cross-device/version bitwise equality is
not promised. Unsupported estimators, feature orders, hooks, layers and unknown
versions fail clearly rather than falling back to opaque serialization.

Regressor feature generation is explicitly CPU float64 in the version-1 bundle
schema; the loader derives that policy from its known feature-order identifier.
Unbundled legacy models retain their original feature path. Legacy CUDA feature
generation used float32, so moving such a calibration to the declared portable
CPU feature path is not qualified for bitwise equality. A stored LUT retains its
exact values. This CC feature policy does not rewrite other legacy stage semantics,
including legacy GC's existing execution choices.

When an older model has no calibration metadata, the bundle retains an empty
metadata dictionary and explicitly marks capture/reference provenance unknown. It
does not invent a measured reference, camera identity or chart capture. Inference
parameters still identify their known numerical operation. The package's existing
legacy input/output interpretation remains the compatibility default.

## Container and validation

The archive is uncompressed `ZIP_STORED`, containing `manifest.json` and canonical
`arrays/aNNNN.npy` members. No member is extracted to the filesystem. Each numeric
array has a declared shape, dtype, byte count and SHA-256; the manifest also has a
canonical content identity. These detect inconsistency/corruption and are **not**
digital signatures or proof of who created the calibration.

The loader checks the ZIP directory bounds before indexing, then rejects duplicate
or undeclared members, traversal paths, links/directories, encrypted/compressed
members, ZIP64/multi-disk/comment/prefix forms, local/directory disagreement and
overlapping/unclaimed bytes. JSON rejects duplicate keys, nonfinite values,
unbounded nesting and unsupported schema fields. Numeric NPY headers are bounded;
object/structured arrays, unexpected dimensions, mismatched payload lengths,
nonfinite values and content-hash failures are rejected before activation.

Arrays are owned immutable buffers. Network connectivity, input/output widths,
normalization, method/basis identity, parameter count and inference probes are
validated. Modern metadata also has the same stage/model/policy consistency checks
as the current pipeline. Fixed loader code chooses all operations; an artifact
cannot import a module or name an arbitrary Python constructor.

Default `BundleLimits` cap the archive and total array bytes at 512 MiB, a single
array at 256 MiB, manifest JSON at 4 MiB and array count at 256. A 6000×4000 float64
FFC gain map occupies 192,000,000 data bytes and fits these limits. Network widths
are bounded to 4096, layers to 256 and expanded parameters to 33,554,432, including
reused-array references. Polynomial regressor degree is limited to 8 before feature
construction; legacy constant degree 0 and at most 33 GC coefficients are supported.
Callers can supply an explicit `BundleLimits` object for their deployment budget.
These byte bounds do not claim a strict process RSS bound or budget full-image
inference workspaces.

Safe integer stage parameters are converted exactly to float64; integers outside
the exactly representable range fail instead of losing precision. Arrays store
floating inference data and do not use NumPy pickle/object loading.

## In-memory inference snapshots

`snapshot_models(models)` makes an owned inference candidate through the same
validated parameter format. `digest_models(models)` identifies canonical metadata
and array contents without hashing executable objects. `dumps_models/loads_models`
provide the same bounded format in memory. This supports a batch capturing one
calibration even while another calibration is fitted or loaded elsewhere.

The format includes all required correction-inference arrays. It does not duplicate
the chart detector's ONNX model: that detector is needed to fit, not to apply an
existing calibration. Its recorded hash/provenance stays in metadata when known.
Every referenced member required for inference must be present and valid.
