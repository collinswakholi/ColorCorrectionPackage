"""Exclusive-window low-level direct timing with frozen P78 source evidence.

Measures public explicit helpers including validation and diagnostic overhead.
Numerical comparisons cover every output, separately from each timed interval.
"""
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

for name in ("NUMBA_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[name] = "8"


def function_source(path,name):
    source = path.read_text(encoding="utf8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name == name)
    return ast.get_source_segment(source,function),function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site",type=Path,required=True)
    parser.add_argument("--reference-site",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repetitions",type=int,default=5)
    args = parser.parse_args()
    os.environ.setdefault("NUMBA_CACHE_DIR",str(args.output.parent/"numba-cache"))
    os.environ.setdefault("MPLBACKEND","Agg")
    sys.dont_write_bytecode = True
    sys.path.insert(0,str(args.site))
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    from threadpoolctl import threadpool_info,threadpool_limits
    from ColorCorrectionPipeline.core import correction,execution,accel
    from ColorCorrectionPipeline.portable_inference import AffinePredictor
    controller = threadpool_limits(limits=8)
    old_path = args.reference_site / "ColorCorrectionPipeline/core/correction.py"
    current_path = Path(correction.__file__)
    old_source,old_function = function_source(old_path,"_predict_direct")
    current_source,_ = function_source(current_path,"_predict_direct")
    if old_source != current_source:
        raise RuntimeError("Default direct code changed: investigate before timing compatibility claims")
    old_transforms = args.reference_site / "ColorCorrectionPipeline/core/transforms.py"
    current_transforms = current_path.with_name("transforms.py")
    old_feature,_ = function_source(old_transforms,"get_poly_features")
    current_feature,_ = function_source(current_transforms,"get_poly_features")
    if old_feature != current_feature:
        raise RuntimeError("Reference feature generator changed")
    scope = dict(vars(correction))
    exec(compile(ast.Module(body=[old_function],type_ignores=[]),str(old_path),"exec"),scope)
    frozen = scope["_predict_direct"]
    data = {"threads":8,"threadpools":threadpool_info(),"site":str(args.site),
        "reference_site":str(args.reference_site),"NumPy":np.__version__,"backend":accel.backend_info(),
        "reference_direct_sha256":hashlib.sha256(old_source.encode()).hexdigest(),
        "reference_direct_source_unchanged":True,"reference_feature_source_unchanged":True,
        "reference_file_sha256":hashlib.sha256(old_path.read_bytes()).hexdigest(),
        "input_seed":1909,"timing_scope":"in-memory public helpers including validation and diagnostics; no I/O, FFC or fitting",
        "note":"Numba warm-up records first call in this process, which may load an existing disk JIT cache; steady times follow warm-up. Not a cold-compilation claim.",
        "runs":[]}
    rng = np.random.default_rng(1909)
    for degree,shape in ((1,(512,512,3)),(1,(1080,1920,3)),(1,(2160,3840,3)),(3,(512,512,3)),(3,(1080,1920,3))):
        count = PolynomialFeatures(degree=degree).fit(np.zeros((1,3))).n_output_features_
        coefficients = rng.normal(0,.025,(3,count))
        coefficients[:,1:4] += np.eye(3)*.8
        model = correction.Regressor_Model()
        model.degree,model.portable_inference,model.use_lut = degree,True,False
        model.model = AffinePredictor(coefficients,np.full(3,.015))
        rgb = rng.random(shape)
        expected = frozen(rgb,model)
        public = correction.predict_(rgb,model)
        if not np.array_equal(public,expected):
            raise RuntimeError("Default public direct predictions changed")
        del public
        record = {"shape":list(shape),"degree":degree,"feature_count":count,
                  "materialized_reference_feature_bytes":int(np.prod(shape[:2])*count*8),
                  "default_public_bitwise_equal":True,"methods":{}}
        for method in ("frozen-p78-direct","numpy","numba"):
            if method == "numba" and not accel.HAS_NUMBA:
                record["methods"][method] = {"unavailable":True}
                continue
            def invoke():
                if method == "frozen-p78-direct": return frozen(rgb,model),{"actual_backend":"reference-cpu-f64"}
                return execution.evaluate_direct(rgb,model,backend=method,allow_fallback=False)
            started = time.perf_counter_ns()
            output,info = invoke()
            first = (time.perf_counter_ns()-started)/1e6
            error = float(np.max(np.abs(output-expected)))
            if not np.allclose(output,expected,rtol=4e-13,atol=4e-13):
                raise RuntimeError("Direct numerical qualification failed: "+method)
            del output
            samples = []
            for _ in range(args.repetitions):
                gc.collect()
                started = time.perf_counter_ns()
                output,info = invoke()
                samples.append((time.perf_counter_ns()-started)/1e6)
                del output
            record["methods"][method] = {"first_full_call_ms":first,"raw_ms":samples,
                "median_ms":statistics.median(samples),"p95_ms":float(np.percentile(samples,95)),
                "all_output_max_abs_error":error,"diagnostics":info}
        data["runs"].append(record)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(data,indent=2),encoding="utf8")
        print(json.dumps({"shape":list(shape),"degree":degree,"median_ms":{key:value.get("median_ms") for key,value in record["methods"].items()}}),flush=True)
        del expected,rgb,model
        gc.collect()
    controller.restore_original_limits()


if __name__ == "__main__":
    main()
