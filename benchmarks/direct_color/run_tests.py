import argparse
import os
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--label',default='full-regression')
parser.add_argument('paths',nargs='*',default=['tests'])
args=parser.parse_args()
root=Path(__file__).resolve().parent
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','MPLBACKEND':'Agg',
    'CCP_PHOTO_DIR':r'C:\Users\Collins\Desktop\Projects\Speedy_CC\CCP_New_UI\test_images'}
for key in ('NUMBA_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): env[key]='8'
for key,directory in [('TEMP','tmp'),('TMP','tmp'),('MPLCONFIGDIR','mpl'),('NUMBA_CACHE_DIR','numba-tests')]:
    (root/directory).mkdir(exist_ok=True); env[key]=str(root/directory)
command=[sys.executable,'-u','-m','pytest',*args.paths,'-o','addopts=','-q','-ra','-p','no:cacheprovider',
    '--basetemp',str(root/('pytest-'+args.label)),'--junitxml',str(root/(args.label+'.xml'))]
with (root/(args.label+'.txt')).open('w',encoding='utf8') as log:
    with subprocess.Popen(command,cwd=r'C:\Users\Collins\Desktop\Projects\Speedy_CC\ColorCorrectionPackage',
            env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace') as process:
        for line in process.stdout:
            log.write(line); log.flush(); print(line,end='',flush=True)
        code=process.wait()
raise SystemExit(code)
