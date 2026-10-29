# Frozen-P9 image validation and export profile

Exclusive run: 2026-10-29 16:24:33–16:26:49 UTC. Three warm repetitions,
eight numerical threads, same previously reviewed 512²/1080p/4K photo PNGs.
No candidate code was timed. All paths use the public APIs and preserve their
complete verification. Garbage collection/profiling/memory probes are outside
timed intervals.

| Operation | 512² median ms | 1080p median ms | 4K median ms |
|---|---:|---:|---:|
| ImageData.from_array float64 | 17.65 | 145.21 | 598.57 |
| ImageData.from_array uint8 | 17.84 | 148.44 | 604.28 |
| Direct ImageData constructor float64 | 11.47 | 92.23 | 367.17 |
| Public typed to_srgb writable copy | 1.40 | 10.36 | 41.27 |
| read_image_data PNG | 382.38 | 564.90 | 1385.68 |
| Strict PNG8 from float64 ndarray | 83.53 | 513.38 | 2110.09 |
| Strict PNG8 from typed float64 | 66.15 | 358.88 | 1399.34 |
| Strict PNG8 from typed uint8 | 67.82 | 356.67 | 1382.21 |
| Strict PNG16 from uint16 ndarray | 104.13 | 620.97 | 2387.03 |
| Strict TIFF64 from float64 ndarray | 77.21 | 456.85 | 2042.55 |

4K cProfile observations (single profiled calls; overlapping cumulative times):

- From-array float64: 566 ms, including 158 ms normalization self time,
  42 ms cast, 362 ms post-init. The two unchanged pixel-data SHA streams cost
  267 ms; immutable byte ownership costs about 40 ms.
- PNG read: 1287 ms. A 7.54 MB encoded input calls
  `stream.read(max_file_bytes+1)` with the default one-GiB cap, allocating close to
  the cap before shrinking. Profiled file read alone costs 259 ms.
- Pillow PNG `getexif()` forces `load()` when no header EXIF exists. The reader
  currently calls it before allocation checks and then decodes again through
  OpenCV. Profiled PNG read has 184 ms Pillow decode plus 157 ms OpenCV decode.
  Removing the duplicate must preserve late metadata and move allocation gates
  ahead of any implicit pixel decoding.
- Strict PNG8 from float64: 1925 ms. Source contract creation takes 560 ms;
  encode 290 ms; raw decoder 221 ms; writer self time 539 ms; five full-frame
  casts total 203 ms. The read-back already compares raw storage samples; it does
  not reconstruct another ImageData. Lossless codec-error computation nevertheless
  builds redundant normalized float64 arrays after exact sample equality proved
  zero error.

Tracemalloc peaks, including NumPy allocations exposed to Python tracing (not all
native codec allocations): float64 from-array 423.0 MB; PNG reader 1073.8 MB;
strict PNG8 from ndarray 1095.0 MB; strict PNG8 typed uint8 895.9 MB.

Raw timings, metadata, source hashes, cProfile text and traced peaks are in
`baseline/results.json`; separate `.prof` files permit deeper inspection.
