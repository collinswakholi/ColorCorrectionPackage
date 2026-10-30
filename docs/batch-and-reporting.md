# Bounded prediction, file outputs and reports

Patches 7–8 add separate APIs while preserving `run()`'s tuple, legacy pickle
methods, `predict_image()`'s default four stages and `predict_images()`'s ordered
list/exception behavior. Use the new APIs when partial failures, cancellation,
controlled retention or non-executable persistence are required.

## Fit/save/load a portable calibration

```python
from ColorCorrectionPipeline import ColorCorrection, Config

config = Config(
    do_ffc=False,
    color_semantics={"version": "explicit-v1"},
    stage_policy="modern-v1", fitting_policy="reproducible-v1",
    CC_kwargs={"degree": 1},
    save=True, save_path="calibration-results", model_format="bundle",
)
pipeline = ColorCorrection()
metrics, stages, failed = pipeline.run(chart_image, name_="camera", config=config)

restored = ColorCorrection().load_bundle("calibration-results/camera_models.ccpb")
# A separate explicit save is also available:
restored.save_bundle("copy.ccpb")
```

`model_format='pickle'` remains the compatibility default. Portable bundle APIs
never guess pickle from a suffix or fall back to it. Unsupported model families
fail explicitly. Loading prepares all parameters and derived context before
activation; failed loading leaves active models/context intact. See
[bundle schema and limits](calibration-bundles.md).

## Selected stages and bounded pointwise workspace

```python
final = restored.predict_image(image, stages="final")["CC"]
selected = restored.predict_image(image, stages=["FFC", "CC"], chunk_pixels=65536)
```

`stages=None`/`'all'` retains the original four keys. `'final'` means only CC;
selected stages retain their pipeline order. Processing stops after the latest
requested stage. Unrequested earlier arrays are released as their successors are
computed, instead of being kept in an output dictionary.

`chunk_pixels` optionally bounds the per-call CC polynomial/NN workspace. New
batch prediction defaults to 65536; ordinary `predict_image` retains its previous
unchunked default. Direct/LUT selection uses the **whole-image pixel count once**,
including short final chunks, so chunking does not silently change inference paths.
Chunked direct linear algebra can have ordinary backend rounding differences;
the qualified tolerance is `1e-12` for the tested float64 paths. Chunking does not
tile spatial FFC, file decoding or every GC temporary buffer.

## Collect per-image outcomes

```python
batch = restored.predict_batch(
    ["capture-a.tiff", "missing.tiff", "capture-a.tiff"],
    stages="final", ordered=True, max_workers=2, max_in_flight=4,
    on_error="continue",
)
for item in batch.items:
    print(item.index, item.status, item.error)
    if item.status == "succeeded":
        corrected = item.images["CC"]
print(batch.summary)
```

Indices identify inputs even when names/objects repeat. `BatchItemResult` contains
input identity, status, one captured calibration identity, selected arrays or export
records, structured error/warnings, timings and input/color provenance. Metadata
mappings are owned/read-only; returned image arrays belong to the result and remain
usable NumPy arrays. `BatchResult.items` intentionally retains collected results.

The batch captures an owned, validated inference snapshot **when the API is called**,
before consuming a generator. A later fit/load or mutation of caller models cannot
change that batch. Configured CC LUT preparation occurs once on that owned snapshot
before workers; a failed preparation selects direct inference with a recorded reason.
Snapshots and this eager preparation cost time/memory, recorded as setup, rather
than mutating the caller's cache or racing multiple initializations.
Pipeline `run`/`load_bundle` synchronize this capture. Callers must synchronize
direct low-level model mutations while the snapshot itself is being copied.

Unknown/unportable estimators cannot enter this new snapshot path. Existing
`predict_images()` remains available for legacy Python-only extension objects.
Unfitted new batch prediction requires `allow_identity=True`; an intentionally
fitted all-disabled calibration is also valid. No identity is reported as a learned
calibration without available metadata.

## Stream without retaining batch history

```python
with restored.iter_predict(
    source_generator, max_workers=2, max_in_flight=4, ordered=False,
    report_path="batch-summary.json",
) as stream:
    for item in stream:
        consume(item)
summary = stream.summary
```

Pending work **and** completed results waiting for ordering share `max_in_flight`.
The source is not materialized, and the stream keeps no result history. NumPy
inputs are copied before submission; immutable `ImageData` is reused. Copies capture
accepted inputs, not future unconsumed list entries. Callers must avoid concurrent
writes during the copy itself. Paths are read when their worker runs; the actual
captured file bytes/hash are recorded. Closed streams release input lists, worker
closures and their owned model/cache buffers while preserving summaries.

Memory is bounded with respect to batch length when the consumer discards emitted
records. It still depends on maximum image dimensions, concurrent/ready records,
selected stages, decoded/normalized source buffers, estimator width/features and
calibration arrays. Bundle/image payload limits are not hard process-RSS limits.
Use appropriate image limits, chunk size, worker and in-flight counts for the host.

## Stop, cancel and callback policy

`on_error='continue'` preserves successes around failed items. `'stop'` stops further
source consumption after a failure is observed and cancels queued work. Inputs
already prefetched/running can finish. No record is invented for an unconsumed
generator element. Known list/tuple totals have an `unsubmitted_count`; unknown
generator totals remain unknown. Source-iteration/close errors have separate fields.

Pass a `threading.Event` as `cancel_event` or call `stream.cancel()`. Cancellation
stops consumption/queued work and is checked before prediction and before export
publication. Native kernels and a publication already in progress cannot be
forcibly interrupted. `close()` waits for bounded running work. Use a context
manager or explicit close when breaking out early; emitted outcomes and abandoned
not-yet-emitted records are counted separately.

`on_progress(item)` runs on the coordinator thread. Callback exceptions add a
warning and increment `callback_errors`; they do not erase successful outputs.
`callback_error='stop'` additionally stops new work. The default is `'record'`.

## File mode and atomic per-item publication

```python
batch = restored.predict_batch(
    source_generator, output_dir="corrected", output_ext=".tiff",
    stages="final", max_workers=2, max_in_flight=4,
)
for item in batch.items:
    if item.status == "succeeded":
        print(item.exports["CC"]["path"])
```

File mode defaults to precision-preserving float TIFF output. It returns no image
arrays. Each invocation creates a unique run directory; each input has an index
directory containing its selected images and `item.json`. This prevents duplicate
basenames and repeated runs from overwriting one another. Images and their manifest
are prepared in a hidden temporary directory, then published with one directory
rename. A failed/cancelled stage set leaves no visible partial item. A cleanup
failure reports its recovery directory without hiding the original export error.

PNG/JPEG quantization requires an explicit choice:

```python
batch = restored.predict_batch(
    images, output_dir="previews", output_ext=".png",
    export_options={"dtype": "uint16", "allow_precision_loss": True},
)
```

The writer receives the actual saved model output encoding. Optional export encoding
conversion is separate from model fitting. See [I/O policies](image-io-policies.md)
for supported storage, ICC/alpha decisions and numerical loss records. A stream's
automatic `batch.json` is an aggregate; per-item manifests retain individual facts.
Publication is per item, not an all-or-nothing transaction across a whole batch or
a power-loss durability guarantee.

## JSON reporting

```python
from ColorCorrectionPipeline import save_report
save_report("collected.json", batch)
pipeline.save_report("calibration-run.json")
```

Reports serialize finite JSON facts and image **descriptors**, never pixel arrays,
opaque estimators or callables. Writes validate a temporary sibling before atomic
replacement. Failed report writing preserves a prior file. An automatic stream
summary write failure remains visible as `summary['report']['status']=='failed'`
without discarding successful item outcomes. Saving a collected `BatchResult` additionally
includes all collected per-item records; streamed summaries deliberately do not
retain that history in RAM.
