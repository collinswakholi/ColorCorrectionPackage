"""Untimed copy-control assertions, separate from performance samples."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parent.parent
OUT = ROOT/"copy-control"
OUT.mkdir(exist_ok=False)
for name in ("NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "8"
os.environ["TEMP"] = os.environ["TMP"] = str(OUT)
os.environ["MPLCONFIGDIR"] = str(OUT/"mpl")
os.environ["NUMBA_CACHE_DIR"] = str(OUT/"numba")
os.environ["MPLBACKEND"] = "Agg"
sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Users\Collins\Desktop\Projects\Speedy_CC\ColorCorrectionPackage")
import numpy as np
from PIL import Image
from ColorCorrectionPipeline import ImageData

rows = []
for size in ("512", "1080p", "4k"):
    pixels = np.asarray(Image.open(AUDIT/"ccp-patches9"/"ui"/"inputs"/(size+".png")).convert("RGB")).astype(np.float64)/255
    typed = ImageData.from_array(pixels)
    copied = typed.to_srgb()
    assert np.array_equal(copied, pixels)
    assert copied.flags.writeable
    assert not np.shares_memory(copied, typed.pixels)
    assert not np.shares_memory(copied, pixels)
    digest = hashlib.sha256(memoryview(copied)).hexdigest()
    saved = typed.pixels[0, 0, 0]
    copied[0, 0, 0] = .125 if saved != .125 else .25
    assert typed.pixels[0, 0, 0] == saved
    rows.append({"size": size, "pixels_exact": True, "writable": True, "owns_independent_copy": True, "mutation_isolated": True, "pixel_payload_sha256": digest})
    del pixels, typed, copied
(OUT/"results.json").write_text(json.dumps(rows, indent=2), encoding="utf8")
print(json.dumps(rows, indent=2))
