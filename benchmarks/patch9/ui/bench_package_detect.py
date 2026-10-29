"""Actual public model detector, isolated CPU/GPU policy and cold/warm scopes."""
import argparse
import importlib.metadata
import sys
from pathlib import Path

from bench_package_api import BASE, measured, stats, write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--site',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,default=7)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.site))
    import cv2
    import ColorCorrectionPipeline as package
    from ColorCorrectionPipeline.detection import detection_context, model_chart_samples, get_detection_diagnostics
    path=BASE/'inputs/chart-1000.png'
    image=cv2.imread(str(path))
    if image is None:raise FileNotFoundError(str(path))
    data=dict(site=str(args.site),imported_from=package.__file__,threads=8,image=str(path),shape=list(image.shape),
        scope='New detection_context each operation; provider sessions reused on warm calls; model_chart_samples(...,1) public API includes one RGB sample percell; file timings include cv2 PNG decode; memory timings exclude decode',
        dependencies={name:importlib.metadata.version(name) for name in ['numpy','onnxruntime','onnxruntime-ep-webgpu']},
        results=[])
    for device in ['cpu','auto']:
        row=dict(device=device)
        def detect(file=False):
            source=cv2.imread(str(path)) if file else image
            with detection_context({'device':device,'cpu_threads':8}):
                result=model_chart_samples(source,1)
                diagnostics=get_detection_diagnostics()
            if result is None:raise RuntimeError('Unexpected MCC compatibility result')
            return dict(sample_shape=list(result.shape),sample_rgb=result,diagnostics=diagnostics)
        try:
            first,first_ms=measured(lambda:detect(file=True))
            row.update(first_file_ms=first_ms,first=first)
            file_times=[];memory_times=[]
            for rep in range(args.repetitions):
                sample,elapsed=measured(lambda:detect(file=True));file_times.append(elapsed)
                _,elapsed=measured(lambda:detect(file=False));memory_times.append(elapsed)
            row.update(file=stats(file_times),inmemory=stats(memory_times),warm=sample)
        except Exception as exc:
            row.update(error_type=type(exc).__name__,error=str(exc),diagnostics=get_detection_diagnostics())
        data['results'].append(row)
        write(args.output/'detection.json',data)
        print(device,row.get('file'),row.get('error'),flush=True)


if __name__=='__main__':main()
