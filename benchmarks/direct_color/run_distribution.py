"""Offline local build and serial isolated-profile qualification; no publishing."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
AUDIT=ROOT.parent
REPO=Path(r'C:\Users\Collins\Desktop\Projects\Speedy_CC\ColorCorrectionPackage')
DIST=ROOT/'candidate-dist'
DIST.mkdir(exist_ok=True)
OUTPUT=ROOT/'distribution'
OUTPUT.mkdir(exist_ok=True)
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','MPLBACKEND':'Agg','PIP_NO_INDEX':'1'}
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): env[key]='1'
env['NUMBA_NUM_THREADS']='2'
for key,directory in [('TEMP','tmp'),('TMP','tmp'),('MPLCONFIGDIR','mpl-dist'),('NUMBA_CACHE_DIR','numba-dist')]:
    (ROOT/directory).mkdir(exist_ok=True); env[key]=str(ROOT/directory)
manifest={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'steps':[]}
def run(label,command,cwd=ROOT):
    result=subprocess.run([str(item) for item in command],cwd=cwd,env=env,capture_output=True,text=True,encoding='utf8',errors='replace')
    (OUTPUT/(label+'.log')).write_text(result.stdout+result.stderr,encoding='utf8')
    manifest['steps'].append({'label':label,'command':[str(item) for item in command],'returncode':result.returncode})
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    if result.returncode:
        print(label,'FAILED',result.stdout[-4000:],result.stderr[-2000:],flush=True); raise SystemExit(result.returncode)
    print(label,'PASS',flush=True)
run('build',[sys.executable,'-I','-m','build','--no-isolation','--wheel','--sdist','--outdir',DIST],REPO)
wheel=next(DIST.glob('*.whl'))
sdist=next(DIST.glob('*.tar.gz'))
run('twine',[AUDIT/'ccp-patches9/distribution/buildenv/Scripts/python.exe','-I','-m','twine','check','--strict',wheel,sdist])
run('archives',[sys.executable,'-I',REPO/'scripts/verify_distribution.py','--dist-dir',DIST,'--output',OUTPUT/'archive-checks.json'])
run('sources',[sys.executable,'-I',REPO/'benchmarks/image_export/fingerprint_sources.py','--wheel',wheel,'--source',REPO,'--output',OUTPUT/'source-fingerprints.json'])
for profile,folder,performance in [('clean38-base','clean38',False),('clean38-performance','clean38-performance',True),('clean312-base','clean312',False),('clean312-performance','clean312-performance',True)]:
    runtime=AUDIT/'ccp-patches9/distribution'/folder
    python=runtime/'Scripts/python.exe'
    run(profile+'-install',[python,'-I','-m','pip','install','--no-index','--no-deps','--force-reinstall',wheel])
    run(profile+'-dependencies',[python,'-I','-m','pip','check'])
    run(profile+'-sources',[python,'-I',REPO/'benchmarks/image_export/fingerprint_sources.py','--wheel',wheel,'--source',runtime/'Lib/site-packages','--output',OUTPUT/(profile+'-sources.json')])
    run(profile+'-full',[python,'-I',REPO/'scripts/verify_installed.py','--output',OUTPUT/(profile+'-full.json'),'--forbid-source',REPO,'--expect-numba','yes' if performance else 'no','--photo',REPO/'images/Image_1.JPG','--legacy-baseline',REPO/'tests/fixtures/legacy_numeric_baseline.json'])
    run(profile+'-older',[python,'-I',AUDIT/'ccp-image-export-opt/scripts/verify_older_bundles.py','--output',OUTPUT/(profile+'-older.json'),'--fixtures',AUDIT/'ccp-patches78/installed-smoke','--legacy-baseline',REPO/'tests/fixtures/legacy_numeric_baseline.json'])
    run(profile+'-direct',[python,'-I',ROOT/'verify_installed_direct.py','--output',OUTPUT/(profile+'-direct.json'),'--expect-numba','yes' if performance else 'no'])
    run(profile+'-image-export',[python,'-I',REPO/'benchmarks/image_export/verify_installed_image_export.py','--output',OUTPUT/(profile+'-image-export.json'),'--fixtures',REPO/'tests/fixtures'])
run('gpu',[AUDIT/'ccp-patches9/distribution/clean312/Scripts/python.exe','-I',REPO/'scripts/verify_installed.py','--output',OUTPUT/'gpu.json','--forbid-source',REPO,'--expect-numba','no','--photo',REPO/'images/Image_1.JPG','--photo-device','auto','--require-photo-gpu','--legacy-baseline',REPO/'tests/fixtures/legacy_numeric_baseline.json'])
manifest['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
manifest['wheel']={'file':str(wheel),'bytes':wheel.stat().st_size,'sha256':hashlib.sha256(wheel.read_bytes()).hexdigest()}
manifest['sdist']={'file':str(sdist),'bytes':sdist.stat().st_size,'sha256':hashlib.sha256(sdist.read_bytes()).hexdigest()}
(OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print('ALL DISTRIBUTION CHECKS PASSED',flush=True)
