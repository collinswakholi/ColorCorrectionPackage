"""Original ONNX detector reference, not a claimed Taichi NN implementation."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
root = Path(__file__).resolve().parent
repo = Path(r'C:\Users\Collins\Desktop\Projects\Speedy_CC\ColorCorrectionPackage')
sys.path.insert(0, str(root / 'baseline_pkg'))
import numpy as np
import cv2
from ColorCorrectionPipeline.detection.charts import _forward, DetectionOptions

p = argparse.ArgumentParser()
p.add_argument('--device', choices=['cpu', 'vulkan'], required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
model = repo / 'ColorCorrectionPipeline/detection/models/chart_detector_1280.onnx'
photo = repo.parent / 'CCP_New_UI/test_images/ColorCard_Big.JPG'
bgr = cv2.imread(str(photo))
opts = DetectionOptions(device=a.device, allow_cpu_fallback=False, cpu_threads=4)
def call():
    return _forward(bgr, model, opts)
t = time.perf_counter_ns()
charts, details = call()
first = (time.perf_counter_ns() - t) / 1e6
for _ in range(3):
    call()
samples = []
for _ in range(11):
    t = time.perf_counter_ns()
    charts, details = call()
    samples.append((time.perf_counter_ns() - t) / 1e6)
result = {'scope': 'unchanged ORT detector, letterbox+ONNX+decode+geometry; excludes disk decode; not Taichi NN inference',
          'device': a.device, 'cpu_fallback_allowed': False, 'original_photo': photo.name,
          'shape': bgr.shape, 'photo_sha256': hashlib.sha256(photo.read_bytes()).hexdigest(),
          'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(), 'cpu_threads': 4,
          'opencv_threads': cv2.getNumThreads(), 'first_call_including_session_init_ms': first,
          'warmups': 3, 'repeats': 11, 'median_ms': float(np.median(samples)),
          'p95_ms': float(np.percentile(samples,95)), 'samples_ms': samples,
          'faces': len(charts), 'execution_details': details}
a.output.write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
