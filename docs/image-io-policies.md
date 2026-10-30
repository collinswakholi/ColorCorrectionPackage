# Explicit image I/O and precision-preserving exports

`read_image_data` and `write_image_data` provide the strict interpretation path.
Existing `read_image`, `read_batch` and `write_image` defaults keep their original
return types and numerical behavior, including the legacy writer's uint8 output.
`read_image(..., strict=True, image_options={...})` routes through the explicit
loader while retaining an array return. The pipeline uses the strict path when
explicit color semantics are selected.

## Loading

```python
from ColorCorrectionPipeline.io import read_image_data

image = read_image_data(
    "capture.tiff",
    color_space="srgb",
    icc_policy="reject",
    apply_orientation=True,
)
```

Uint8/uint16 RGB samples retain their precision; normalized float32/float64 RGB
TIFF is read through tifffile. Orientation is applied once and its original value
is retained. File metadata and pixel decoding use the same captured byte buffer,
whose SHA-256 is recorded. TIFF uses native RGB/gray samples rather than OpenCV's
potentially lossy float RGB LogLuv path. Unsupported TIFF photometric/layouts and
nonfinite/out-of-contract float samples fail rather than being relabeled/clipped.

Ordinary files default to the declared sRGB assumption. Exports made by this
library carry versioned encoding metadata. With `color_space=None` (the loader
default), that metadata is honored. An explicit conflicting `color_space` fails;
use the correct input interpretation and request a deliberate conversion later.
This does not infer arbitrary input primaries or transfer functions from every
possible image metadata convention.

Preprocessing choices are explicit:

- `alpha_policy='reject'` is default. `'drop'` discards straight alpha deliberately.
  `'composite'` uses `alpha_background=(R,G,B)` and linear-light compositing. The
  background uses the declared input encoding. Original decoded dtype/shape and
  the decision are recorded. Associated/premultiplied alpha requires external
  normalization and is rejected rather than guessed.
- `grayscale_policy='reject'` is default. `'replicate'` copies gray into RGB; TIFF
  white-is-zero interpretation is honored. This is not spectral/color recovery.
- Multiframe files require an explicit zero-based `frame_index`; unsupported
  high-precision non-TIFF frame selection fails instead of reducing to uint8.
- `icc_policy='reject'` is default. `'assume-declared'` records the caller's
  interpretation without a color transform. `'convert-srgb'` explicitly invokes
  Pillow ImageCms/LCMS for supported uint8 RGB or CMYK profiles, using relative
  colorimetric rendering intent. High-precision/profile combinations outside
  that path fail without a uint8 downgrade. Invalid/rejected profiles fail before
  native pixel decoding. Source profile hash and actual transform are recorded.

Allocation preflight defaults to `max_pixels=100_000_000` and
`max_decoded_bytes=1_073_741_824`, with `max_file_bytes=1_073_741_824` for the captured
encoded payload. File size is checked before opening and reading is bounded even
if a concurrent writer grows the file. Dimensions/native sample size are inspected before
pixel decoding. The byte limit conservatively estimates decoded source samples plus canonical
float64 RGB, not just compressed file size. Decoder workspaces, metadata/file
buffers and later stage intermediates require additional memory; this is not a
hard process RSS limit. Limits are positive integer options and can be set to the
application's capture requirements.

## Exporting

```python
from ColorCorrectionPipeline.io import write_image_data

# Preserve a uint16 ImageData source without an implicit uint8 conversion.
record = write_image_data("corrected.png", image16)

# Pipeline float outputs: storage quantization is an explicit decision.
record = write_image_data(
    "corrected.png", corrected_float_rgb,
    input_encoding="srgb", dtype="uint16", allow_precision_loss=True,
)

# Scientific normalized floats: preserve their selected storage precision.
record = write_image_data("corrected.tiff", corrected_float_rgb)
facts = record.to_dict()
```

| Format | Supported output storage | Encoding declaration |
| --- | --- | --- |
| PNG | uint8/uint16 | Native sRGB or linear gAMA/chromaticity tags, plus project encoding text |
| TIFF | uint8/uint16/float32/float64 | Uncompressed RGB IEEE samples, project encoding description, identity orientation |
| JPEG/WebP preview | uint8 | Requires sRGB output interpretation; codec loss is reported |

`dtype='preserve'` uses the source storage dtype; float16 is promoted losslessly to
float32 TIFF. A float source cannot silently become integer PNG/JPEG. Reducing
storage precision or quantizing floats to integers requires both an explicit dtype
and `allow_precision_loss=True`. Integer output after an encoding conversion also
requires that consent, even when its storage bit depth is unchanged: converted
values must be requantized. Same-encoding integer exports preserve their samples
without this additional gate. Choosing JPEG/WebP remains a lossy format choice
even for uint8; the consent switch governs precision reduction, not a promise of
codec losslessness. Linear JPEG/WebP is rejected until output is explicitly
converted to sRGB.

Typed `ImageData` owns encoding/range/channel declarations. Conflicting export
reinterpretation fails. Raw arrays may declare `input_encoding`, `channel_order`
and `value_range`; they must satisfy the same normalized RGB contract. Optional
`output_encoding` converts between sRGB and linear-sRGB only. Pending orientation
is normalized once before fresh encoding, and obsolete source ICC/EXIF is not
copied blindly into changed pixels.

PNG color metadata follows the [W3C PNG color-space specification](https://www.w3.org/TR/png/#11cHRM).
TIFF project descriptions identify encoding for this library and reports; they
do not guarantee that every external TIFF viewer honors linear-sRGB descriptions.
Use the returned export facts when exchanging scientific files, and qualify the
receiving application's color policy.

Encoding finishes in a temporary sibling file, is flushed and read back, and must
match its claimed dtype/shape/format. Lossless formats reproduce quantized sample
values exactly. `overwrite=False` is default: Windows no-clobber rename or POSIX
hard-link publication prevents races from replacing another writer's file.
`overwrite=True` uses atomic replacement after validation. Failures preserve an
existing destination and remove temporary files. This is local filesystem atomic
publication, not a promise of power-loss durability or multi-file transactions.

The immutable `ExportRecord` retains output path/hash/bytes, actual format/dtype,
shape/encoding, input encoding/storage/range/channel/orientation/source identities,
precision-loss decision, quantization error and measured codec difference. It
retains no image arrays. `to_dict()` returns JSON-plain copies. Batch file mode
stores these records in its report rather than silently emitting an unpaired
sidecar from the image writer.


## Optional fast-lossless PNG representation

The default `png_profile="compatible"` retains the established PNG encoder and
quality/compression behavior. Explicit `png_profile="fast-lossless"` selects RGB
PNG8/PNG16 with filter None and compression level zero; it overrides the PNG
quality/compression setting. Source quantization and precision-loss consent are
unchanged. The codec is lossless relative to the selected integer samples.
Encoded bytes, SHA256 and file size differ intentionally; all strict fsync,
read-back, decoded sample/metadata verification and atomic publication checks stay
active. Other formats reject this PNG-only option. Batch `export_options` accepts
the same profile. See [performance measurements](performance.md).

Eligible large integer normalization/quantization uses qualified optional CPU
Numba arithmetic, with NumPy fallback and exact operation order. Images below
8 MiB canonical float64 size retain NumPy. Model execution's `backend` option
controls its own kernels; this I/O selection independently preserves the same
pixels and error facts. Background workers use serial GIL-releasing I/O kernels,
avoiding nested native worker pools. No process-wide thread settings are changed.

`ImageData.pixels` provides a fresh read-only ndarray header over immutable
storage. Local shape/dtype/stride edits to a returned view cannot change the
canonical image or invalidate its recorded digests. To declare a new geometry,
construct another ImageData explicitly from the desired view.
