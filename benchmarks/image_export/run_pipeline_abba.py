"""Sequential fresh-process timing driver; no concurrent benchmark workers."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parent.parent
P9 = AUDIT / "ccp-patches9"
SITES = {"A": P9/"wheel-site", "B": Path(r"C:\Users\Collins\Desktop\Projects\Speedy_CC\ColorCorrectionPackage")}
OUTPUT = ROOT/"pipeline-abba"
OUTPUT.mkdir(exist_ok=False)
manifest = {"order": [], "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
for mode in ("direct", "relaxed"):
    for label in ("A1", "B1", "B2", "A2"):
        destination = OUTPUT/(mode+"-"+label)
        command = [sys.executable, "-u", str(ROOT/"bench_pipeline_io.py"),
                   "--site", str(SITES[label[0]]), "--models", str(P9/"ui"/"benchmark-p78"),
                   "--inputs", str(P9/"ui"/"inputs"), "--output", str(destination),
                   "--sizes", "4k", "--modes", mode, "--repetitions", "3", "--file-only"]
        print(mode, label, "START", datetime.datetime.now(datetime.timezone.utc).isoformat(), flush=True)
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, text=True, capture_output=True)
        log = OUTPUT/(mode+"-"+label+".log")
        log.write_text(completed.stdout+completed.stderr, encoding="utf8")
        manifest["order"].append({"mode": mode, "label": label, "command": command,
            "seconds": time.perf_counter()-started, "returncode": completed.returncode})
        (OUTPUT/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
        print(mode, label, "END", completed.returncode, completed.stdout.strip(), flush=True)
        if completed.returncode:
            print(completed.stderr, flush=True)
            raise SystemExit(completed.returncode)
manifest["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
(OUTPUT/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
