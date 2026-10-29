"""Exclusive CPU window: frozen/candidate ABBA, then actual UI Python API."""
import datetime
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
TASK=ROOT.parent
AUDIT=TASK.parent
REPO=Path(r'C:\Users\Collins\Desktop\Projects\Speedy_CC\ColorCorrectionPackage')
UI=REPO.parent/'CCP_New_UI'
NATIVE=UI/'_implementation/ui-docs-20261026/sandbox/harness/candidate-win-x64-v3'
parser=argparse.ArgumentParser()
parser.add_argument('--label',default='measured')
parser.add_argument('--file-only',action='store_true')
args=parser.parse_args()
OUTPUT=ROOT/args.label
OUTPUT.mkdir(exist_ok=False)
commands=[]
for label,site in [('A1',TASK/'baseline-site'),('B1',REPO),('B2',REPO),('A2',TASK/'baseline-site')]:
    commands.append((label,[sys.executable,'-u',str(ROOT/'bench_direct.py'),'--site',str(site),
        '--models',str(AUDIT/'ccp-patches9/ui/benchmark-p78'),'--inputs',str(AUDIT/'ccp-patches9/ui/inputs'),
        '--output',str(OUTPUT/label),'--modes','direct','--repetitions','3']))
    if args.file_only: commands[-1][1].append('--file-only')
commands.append(('UI',[sys.executable,'-u',str(ROOT/'bench_ui.py'),'--ui',str(UI),'--native',str(NATIVE),
    '--prior-benchmark',str(AUDIT/'ccp-patches9/ui/benchmark-ui/benchmark.json'),
    '--inputs',str(AUDIT/'ccp-patches9/ui/inputs'),'--output',str(OUTPUT/'UI'),'--repetitions','7']))
manifest={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'workers':[]}
for label,command in commands:
    print(label,'START',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
    started=time.perf_counter()
    result=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,encoding='utf8',errors='replace')
    (OUTPUT/(label+'.log')).write_text(result.stdout+result.stderr,encoding='utf8')
    manifest['workers'].append({'label':label,'command':command,'seconds':time.perf_counter()-started,'returncode':result.returncode})
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(label,'END',result.returncode,result.stdout,flush=True)
    if result.returncode:
        print(result.stderr,flush=True); raise SystemExit(result.returncode)
manifest['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
