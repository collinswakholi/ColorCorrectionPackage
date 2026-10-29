"""Sequential final candidate and contemporary frozen-control workers."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parent.parent
P9 = AUDIT/"ccp-patches9"
CLONE = Path(r"C:\Users\Collins\Desktop\Projects\Speedy_CC\ColorCorrectionPackage")
OUTPUT = ROOT/"final-revision"
OUTPUT.mkdir(exist_ok=False)
commands = [("components", [sys.executable, "-u", str(ROOT/"bench_image_io.py"), "--site", str(CLONE),
    "--inputs", str(P9/"ui"/"inputs"), "--output", str(OUTPUT/"components"), "--profile", "--memory"])]
for label, site, mode in (("candidate-direct", CLONE, "direct"), ("candidate-relaxed", CLONE, "relaxed"), ("control-relaxed", P9/"wheel-site", "relaxed")):
    commands.append((label, [sys.executable, "-u", str(ROOT/"bench_pipeline_io.py"), "--site", str(site),
        "--models", str(P9/"ui"/"benchmark-p78"), "--inputs", str(P9/"ui"/"inputs"), "--output", str(OUTPUT/label),
        "--sizes", "4k", "--modes", mode, "--repetitions", "3", "--file-only"]))
manifest = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "order": []}
for label, command in commands:
    print(label, "START", datetime.datetime.now(datetime.timezone.utc).isoformat(), flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, text=True, capture_output=True)
    (OUTPUT/(label+".log")).write_text(result.stdout+result.stderr, encoding="utf8")
    manifest["order"].append({"label": label, "command": command, "seconds": time.perf_counter()-started, "returncode": result.returncode})
    (OUTPUT/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(label, "END", result.returncode, result.stdout.strip(), flush=True)
    if result.returncode:
        print(result.stderr, flush=True)
        raise SystemExit(result.returncode)
manifest["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
(OUTPUT/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
