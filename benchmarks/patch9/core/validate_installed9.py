"""Final installed P9 qualification on genuine earlier calibration artifacts.

Run only after the coordinated timing window. No repository writes/publishing.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent.parent
    parser.add_argument("--site",type=Path,default=base/"wheel-site")
    parser.add_argument("--dist",type=Path,default=base/"distribution/final-dist")
    parser.add_argument("--output",type=Path,default=base/"installed-smoke")
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND","Agg")
    os.environ.setdefault("NUMBA_NUM_THREADS","8")
    os.environ.setdefault("NUMBA_CACHE_DIR",str(base/"installed-numba-cache"))
    sys.dont_write_bytecode = True
    sys.path.insert(0,str(args.site))
    import numpy as np
    import ColorCorrectionPipeline as ccp
    from ColorCorrectionPipeline.io import read_image_data,write_image_data

    audit = base.parent/"ccp-plan-audit"
    source = Path(r"C:\Users\Collins\Desktop\Projects\Speedy_CC\ColorCorrectionPackage")
    installed = args.site.resolve()
    assert Path(ccp.__file__).resolve().is_relative_to(installed),ccp.__file__
    fingerprints = {}
    for path in (source/"ColorCorrectionPipeline").rglob("*.py"):
        assert path.read_bytes() == (installed/path.relative_to(source)).read_bytes(),str(path)
        fingerprints[path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    wheels,sdists = sorted(args.dist.glob("*.whl")),sorted(args.dist.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1,"Use a final-dist with precisely one wheel and sdist"
    wheel,sdist = wheels[0],sdists[0]
    with zipfile.ZipFile(wheel) as package:
        for name in ("bundle.py","portable_inference.py","prediction.py","batch.py","results.py","reporting.py", "acceleration.py", "core/execution.py","core/lut_cache.py"):
            assert "ColorCorrectionPipeline/"+name in package.namelist(),name
        metadata_name = next(name for name in package.namelist() if name.endswith(".dist-info/METADATA"))
        assert b"tifffile>=" in package.read(metadata_name)
        model_hash = hashlib.sha256(package.read("ColorCorrectionPipeline/detection/models/chart_detector_1280.onnx")).hexdigest()
        assert model_hash == "1d2e9cfca6c13922ad16abeb196b0ad3071778c0067bc9ea95c83b0e76ef2240"
    directory = args.output
    directory.mkdir(parents=True,exist_ok=True)
    previous = base.parent/"ccp-patches56/installed-smoke"
    image = np.load(previous/"input.npy")
    np.save(directory/"input.npy",image)
    # Historical CPU float32 output can depend on Torch's thread context. Pin a
    # frozen P78 runtime to the same eight-thread environment for strict default
    # compatibility, while retaining the original stored-array comparison too.
    matched = directory/"matched-p78"
    matched.mkdir(exist_ok=True)
    baseline_worker = r'''
import sys,json
from pathlib import Path
import numpy as np,torch
import ColorCorrectionPipeline as ccp
p=Path(sys.argv[1]);out=Path(sys.argv[2]);x=np.load(p/'input.npy');report={}
for method in ('linear','pls','nn','custom','conv'):
    cc=ccp.ColorCorrection();cc.models.load(p,method+'_models')
    actual=cc.predict_image(x)['CC'];historical=np.load(p/(method+'-expected.npy'))
    np.save(out/(method+'-expected.npy'),actual)
    report[method]={'historical_bitwise_equal':bool(np.array_equal(actual,historical)),
        'historical_max_difference':float(np.max(np.abs(actual-historical))),
        'historical_mismatches':int(np.count_nonzero(actual!=historical))}
cc=ccp.ColorCorrection();cc.models.load(p,'custom_models');historical=np.load(p/'custom-expected.npy')
probes={}
for count in (1,2,4,8):
    torch.set_num_threads(count)
    actual=cc.predict_image(x)['CC']
    probes[str(count)]={'historical_bitwise_equal':bool(np.array_equal(actual,historical)),
        'historical_max_difference':float(np.max(np.abs(actual-historical))),
        'historical_mismatches':int(np.count_nonzero(actual!=historical))}
(out/'reference.json').write_text(json.dumps({'package':ccp.__file__,'matched_threads':8,
    'families':report,'custom_torch_thread_probe':probes},indent=2),encoding='utf8')
'''
    reference_site = base.parent/"ccp-patches78/wheel-site"
    subprocess.run([sys.executable,"-B","-c",baseline_worker,str(previous),str(matched)],cwd=base,
        env=dict(os.environ,PYTHONPATH=str(reference_site)),check=True,timeout=180,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    matched_context = json.loads((matched/"reference.json").read_text(encoding="utf8"))
    families = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for method in ("linear","pls","nn","custom","conv"):
            original = ccp.ColorCorrection()
            original.models.load(previous,method+"_models")
            expected = original.predict_image(image)["CC"]
            historical_expected = np.load(previous/(method+"-expected.npy"))
            np.testing.assert_array_equal(expected,np.load(matched/(method+"-expected.npy")))
            path = directory/(method+".ccpb")
            original.save_bundle(path)
            loaded = ccp.ColorCorrection().load_bundle(path)
            actual = loaded.predict_image(image)["CC"]
            np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-12)
            np.save(directory/(method+"-expected.npy"),actual)
            batch = loaded.predict_batch([image,directory/"missing.png",image],ordered=True,
                max_workers=2,max_in_flight=2,chunk_pixels=65536)
            assert [item.status for item in batch.items] == ["succeeded","failed","succeeded"]
            tolerance = 2e-6 if method == "custom" else 2e-12
            for item in (batch.items[0],batch.items[2]):
                np.testing.assert_allclose(item.images["CC"],actual,atol=tolerance,rtol=tolerance)
            # Default prediction can intentionally use the saved CC LUT. Its
            # historical parity check above is separate from direct execution.
            reference_pipeline = ccp.ColorCorrection().load_bundle(path)
            if reference_pipeline.models.model_cc[2] == "ours":
                reference_model = reference_pipeline.models.model_cc[0]
                reference_model.use_lut = False
                reference_model.clear_lut_cache()
                reference_model._lut_predict_fn = None
            reference_direct_expected = reference_pipeline.predict_image(image)["CC"]
            with loaded.prepare_execution(dict(mode="direct",backend="auto",validation_samples=64)) as prepared:
                accelerated = prepared.predict_image(image)
                np.testing.assert_allclose(accelerated.images["CC"],reference_direct_expected,atol=tolerance,rtol=tolerance)
                proof = prepared.preparation["qualification"]
            families[method] = {"p56_pickle_predictions_bitwise_equal_to_matched_p78":True,
                "p56_pickle_predictions_bitwise_equal_to_historical_array":bool(np.array_equal(expected,historical_expected)),
                "p56_historical_max_difference":float(np.max(np.abs(expected-historical_expected))),
                "p56_historical_mismatches":int(np.count_nonzero(expected!=historical_expected)),
                "portable_reload_equal":bool(np.array_equal(actual,expected)),
                "portable_reload_max_difference":float(np.max(np.abs(actual.astype(float)-expected.astype(float)))),
                "portable_reload_tolerance":2e-12,"mixed_batch_passed":True,"chunked_tolerance":tolerance,
                "prepared_direct_passed":True,"prepared_direct_reference":"fresh portable bundle with CC use_lut=False; no CC interpolation",
                "prepared_direct_max_difference":float(np.max(np.abs(accelerated.images["CC"]-reference_direct_expected))),
                "prepared_direct_tolerance":tolerance,"prepared_direct_qualification":proof}
    worker = r'''
import sys,numpy as np
from pathlib import Path
from ColorCorrectionPipeline import ColorCorrection
p=Path(sys.argv[1]);x=np.load(p/'input.npy')
for method in ('linear','pls','nn','custom','conv'):
    model=ColorCorrection().load_bundle(p/(method+'.ccpb'))
    np.testing.assert_array_equal(model.predict_image(x)['CC'],np.load(p/(method+'-expected.npy')))
'''
    subprocess.run([sys.executable,"-B","-c",worker,str(directory)],cwd=base,
        env=dict(os.environ,PYTHONPATH=str(installed)),check=True,timeout=180,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    for result in families.values():
        result["fresh_process_bundle_predictions_bitwise_equal"] = True
    with contextlib.redirect_stdout(io.StringIO()):
        fitted = ccp.ColorCorrection()
        config = ccp.Config(do_ffc=False,color_semantics={"version":"explicit-v1"},
            stage_policy="modern-v1",fitting_policy="reproducible-v1",CC_kwargs={"degree":1},
            save=True,save_path=str(directory),model_format="bundle")
        _,stages,failed = fitted.run(ccp.ImageData.from_array(image),name_="installed",config=config)
        assert not failed,fitted.last_run
        detected = next(row for row in fitted.last_run["detection"] if row.get("status") == "detected")
        assert detected["provider"] == "WebGpuExecutionProvider",detected
        assert detected["execution_verified"] is True and detected["node_provider_counts"].get("CPUExecutionProvider",0) == 0,detected
        reloaded = ccp.ColorCorrection().load_bundle(directory/"installed_models.ccpb")
        np.testing.assert_allclose(reloaded.predict_image(image)["CC"],stages["installed_CC"],atol=2e-12,rtol=2e-12)
        exports = reloaded.predict_batch([image,image],output_dir=directory/"exports",max_workers=2)
        assert exports.summary["succeeded"] == 2
        expected = reloaded.predict_image(image)["CC"]
        for item in exports.items:
            assert not item.images
            np.testing.assert_allclose(read_image_data(item.exports["CC"]["path"]).pixels,expected,atol=2e-12,rtol=2e-12)
    integer = np.arange(180,dtype=np.uint16).reshape(6,10,3)*307
    record = write_image_data(directory/"uint16.png",ccp.ImageData.from_array(integer),overwrite=True)
    np.testing.assert_array_equal(read_image_data(record.path).pixels,integer/65535)
    legacy = ccp.ColorCorrection()
    legacy.models.load(audit/"wheel-final-smoke","wheel_models")
    legacy_input = np.random.default_rng(81).uniform(.01,.99,(80,80,3))
    np.save(directory/"legacy-input.npy",legacy_input)
    old_worker = r'''
import sys,numpy as np
from pathlib import Path
from ColorCorrectionPipeline import ColorCorrection
a=Path(sys.argv[1]);p=Path(sys.argv[2]);x=np.load(p/'legacy-input.npy')
cc=ColorCorrection();cc.models.load(a/'wheel-final-smoke','wheel_models')
np.save(p/'legacy-large.npy',cc.predict_image(x)['CC'])
np.save(p/'legacy-small.npy',cc.predict_image(x[:4,:4])['CC'])
'''
    subprocess.run([sys.executable,"-B","-c",old_worker,str(audit),str(directory)],cwd=base,
        env=dict(os.environ,PYTHONPATH=str(audit/"wheel-final-site")),check=True,timeout=180,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    with contextlib.redirect_stdout(io.StringIO()):
        np.testing.assert_array_equal(legacy.predict_image(legacy_input)["CC"],np.load(directory/"legacy-large.npy"))
        np.testing.assert_array_equal(legacy.predict_image(legacy_input[:4,:4])["CC"],np.load(directory/"legacy-small.npy"))
    report = {"installed_sources_match_clone":True,"final_python_fingerprints":fingerprints,
        "families":families,"matched_context_reference":matched_context,
        "actual_new_gpu_fit_bundle_reload_and_file_batch_passed":True,
        "gpu_execution":{key:detected[key] for key in ("provider","graphics_backend","execution_verified","node_provider_counts")},
        "uint16_png_exact_sample_roundtrip":True,"legacy_p02_saved_model_predictions_bitwise_equal":True,
        "model_sha256":model_hash,"wheel":{"path":str(wheel),"bytes":wheel.stat().st_size,"sha256":hashlib.sha256(wheel.read_bytes()).hexdigest()},
        "sdist":{"path":str(sdist),"bytes":sdist.stat().st_size,"sha256":hashlib.sha256(sdist.read_bytes()).hexdigest()},
        "qualification":"Windows installed target with audited scientific dependencies; clean dependency and cross-platform qualification are separate distribution evidence"}
    (directory/"validation.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
