"""Non-executable bundle family parity, resource limits and corruption cases."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import stat
import struct
import subprocess
import sys
import warnings
import zipfile

import numpy as np
import pytest

from ColorCorrectionPipeline import ColorCorrection, MyModels
from ColorCorrectionPipeline.core.correction import fit_model, predict_
from ColorCorrectionPipeline.fitting import fit_observations
from ColorCorrectionPipeline.stage_policies import MonotonicToneModel
from ColorCorrectionPipeline.bundle import (BundleError, BundleLimits, dumps_models, loads_models,
    save_bundle, load_bundle, digest_models, snapshot_models)
import ColorCorrectionPipeline.bundle as bundle


@pytest.fixture(scope="module", autouse=True)
def bounded_torch_threads():
    import torch
    previous = torch.get_num_threads()
    torch.set_num_threads(min(2, previous))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def paired():
    x = np.random.default_rng(811).uniform(.05, .95, (32, 3))
    return x, x * np.array([.88, .92, .85]) + .02


def calibration(method="linear", modern=False, built=False, bn=False):
    x, y = paired()
    parameters = {"mtd": method, "degree": 1, "use_lut": True,
                  "lut_grid_size": 5, "lut_min_pixels": 10, "lazy_lut": True}
    if method in ("nn", "custom"):
        parameters.update(hidden_layers=[5], max_iterations=20 if method == "nn" and not modern else 4, batch_size=8, patience=2)
    if method == "custom":
        parameters.update(dropout_rate=.2, use_batch_norm=bn)
    if method == "pls":
        parameters.update(ncomp=2, max_iterations=100)
    if modern:
        model = fit_observations(x, y, ["cell%d" % i for i in range(len(x))], parameters, "reproducible-v1")
    else:
        fitting_parameters = dict(parameters, mtd="linear") if method == "linearCC" else parameters
        model = fit_model(x, y, fitting_parameters)
        if method == "linearCC":
            model.mtd = "linearCC"  # known older fitted-container spelling
    if built:
        predict_(x.reshape(4, 8, 3), model)
    result = MyModels()
    result.model_ffc = np.linspace(.9, 1.2, 32).reshape(4, 8)
    result.model_gc = np.array([1., 0.])
    result.model_wb = np.diag([.95, .92, .99])
    result.model_cc = (model, parameters, "ours")
    return result


def unpack(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members.pop("manifest.json"))
    return manifest, members


def repack(manifest, members, *, compression=zipfile.ZIP_STORED):
    manifest = copy.deepcopy(manifest)
    manifest.pop("content_sha256", None)
    manifest["content_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=True,
                   allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True, separators=(",", ":")))
        for name, value in members.items():
            archive.writestr(name, value)
    return output.getvalue()


def change_member(manifest, members, index, raw, *, dtype=None, shape=None):
    record = manifest["arrays"][index]
    members[record["path"]] = raw
    record.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    if dtype is not None:
        record["dtype"] = dtype
    if shape is not None:
        record["shape"] = shape


@pytest.mark.parametrize("method", ["linear", "linearCC", "pls", "nn", "custom"])
@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("built", [False, True])
def test_known_families_preserve_direct_and_lut_predictions(method, modern, built, tmp_path):
    models = calibration(method, modern, built)
    x, _ = paired()
    expected_direct = predict_(x[:4], models.model_cc[0])
    destination = tmp_path / "calibration.ccpb"
    save_bundle(models, destination)
    result = load_bundle(destination)
    assert result.model_cc[0].portable_inference is True
    assert (result.model_cc[0]._lut is not None) == built
    assert result.metadata == {}
    assert "unknown" in result.bundle_interpretation
    np.testing.assert_array_equal(result.model_ffc, models.model_ffc)
    np.testing.assert_array_equal(result.model_gc, models.model_gc)
    np.testing.assert_array_equal(result.model_wb, models.model_wb)
    np.testing.assert_allclose(predict_(x[:4], result.model_cc[0]), expected_direct, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(predict_(x.reshape(4, 8, 3), result.model_cc[0]),
                               predict_(x.reshape(4, 8, 3), models.model_cc[0]), atol=2e-12, rtol=2e-12)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("modern", [False, True])
def test_torch_batchnorm_and_dropout_eval_parameters_preserved(modern):
    models = calibration("custom", modern, bn=True)
    x, _ = paired()
    result = loads_models(dumps_models(models))
    np.testing.assert_array_equal(predict_(x[:4], result.model_cc[0]), predict_(x[:4], models.model_cc[0]))


@pytest.mark.parametrize("root", [False, True, None])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_conventional_feature_basis_preserved(root, degree):
    import colour
    x, y = paired()
    options = {"method": "Finlayson 2015", "degree": degree, "root_polynomial_expansion": root, "terms": None}
    models = MyModels()
    matrix = colour.characterisation.matrix_colour_correction(x, y, **options)
    models.model_cc = (matrix, options, "conv")
    restored = loads_models(dumps_models(models))
    before = colour.characterisation.apply_matrix_colour_correction(x, matrix, **options)
    after = colour.characterisation.apply_matrix_colour_correction(x, restored.model_cc[0], **restored.model_cc[1])
    np.testing.assert_array_equal(after, before)


def test_tone_endpoints_and_owned_arrays_survive_bundle():
    models = MyModels()
    models.model_gc = MonotonicToneModel(np.array([0., 10, 40, 80, 100]), np.array([0., 20, 50, 90, 100]))
    # A metadata-free low-level model has no capture facts to fabricate.
    restored = loads_models(dumps_models(models))
    np.testing.assert_array_equal(restored.model_gc.predict(np.linspace(0, 100, 101)), models.model_gc.predict(np.linspace(0, 100, 101)))
    with pytest.raises(ValueError):
        restored.model_gc.measured_knots.setflags(write=True)


def test_identity_empty_container_is_valid_and_deterministic():
    models = MyModels()
    assert dumps_models(models) == dumps_models(models)
    restored = snapshot_models(models)
    assert not restored.has_models()
    assert digest_models(models) == restored.bundle_digest == digest_models(restored)


def test_snapshot_owns_all_arrays_and_original_mutation_does_not_change_prediction():
    original = calibration()
    x, _ = paired()
    snapshot = snapshot_models(original)
    expected = predict_(x[:4], snapshot.model_cc[0])
    original.model_ffc[:] = 9
    original.model_cc[0].model.coef_[:] = 0
    np.testing.assert_array_equal(predict_(x[:4], snapshot.model_cc[0]), expected)
    with pytest.raises(ValueError):
        snapshot.model_ffc.setflags(write=True)
    with pytest.raises(ValueError):
        snapshot.model_cc[0].model.coefficients.setflags(write=True)


@pytest.mark.parametrize("activation", ["identity", "logistic", "tanh", "relu"])
def test_sklearn_mlp_known_activation_parity(activation):
    from sklearn.neural_network import MLPRegressor
    models = calibration("linear")
    x, y = paired()
    features = np.c_[np.ones(len(x)), x]
    estimator = MLPRegressor(hidden_layer_sizes=(4,), activation=activation, max_iter=3, random_state=2).fit(features, y)
    models.model_cc[0].model = estimator
    models.model_cc[0].mtd = "nn"
    restored = loads_models(dumps_models(models))
    np.testing.assert_array_equal(predict_(x[:4], restored.model_cc[0]), predict_(x[:4], models.model_cc[0]))


@pytest.mark.parametrize("mutation", ["format", "version", "unknown-root", "unknown-stage", "unknown-estimator", "feature-order", "wrong-coef-shape", "method-mismatch", "bad-lut-grid", "bad-lut-type", "bad-array-path", "bad-array-dtype", "huge-array-shape", "bad-identity", "duplicate-record"])
def test_manifest_corruptions_rejected_even_with_recomputed_identity(mutation):
    manifest, members = unpack(dumps_models(calibration()))
    cc = manifest["stages"]["CC"]
    if mutation == "format": manifest["format"] = "pickle"
    elif mutation == "version": manifest["version"] = 999
    elif mutation == "unknown-root": manifest["load_class"] = "os.system"
    elif mutation == "unknown-stage": manifest["stages"]["EXEC"] = "malicious"
    elif mutation == "unknown-estimator": cc["estimator"]["kind"] = "arbitrary-python"
    elif mutation == "feature-order": cc["feature_order"] = "unknown"
    elif mutation == "wrong-coef-shape": cc["attributes"]["degree"] = 2
    elif mutation == "method-mismatch": cc["attributes"]["mtd"] = "custom"
    elif mutation == "bad-lut-grid": cc["attributes"]["_lut_grid_size"] = 0
    elif mutation == "bad-lut-type": cc["attributes"]["use_lut"] = 1
    elif mutation == "bad-array-path": manifest["arrays"][0]["path"] = "../array.npy"
    elif mutation == "bad-array-dtype": manifest["arrays"][0]["dtype"] = "|O"
    elif mutation == "huge-array-shape": manifest["arrays"][0]["shape"] = [10**12, 10**12]
    elif mutation == "bad-identity":
        raw = repack(manifest, members).replace(b'"content_sha256":"', b'"content_sha256":"0', 1)
        with pytest.raises(BundleError): loads_models(raw)
        return
    else: manifest["arrays"].append(copy.deepcopy(manifest["arrays"][0]))
    with pytest.raises(BundleError):
        loads_models(repack(manifest, members))


@pytest.mark.parametrize("kind", ["nan", "object", "wrong-shape", "trailing", "hash", "header-size"])
def test_numeric_payload_corruptions_never_use_pickle(kind, monkeypatch):
    manifest, members = unpack(dumps_models(calibration()))
    original = members[manifest["arrays"][0]["path"]]
    if kind == "hash":
        members[manifest["arrays"][0]["path"]] = original[:-1] + bytes([original[-1] ^ 1])
    elif kind == "trailing":
        change_member(manifest, members, 0, original + b"unclaimed")
    elif kind == "header-size":
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {"descr": "<f8", "fortran_order": False, "shape": (10**100, 1)})
        change_member(manifest, members, 0, stream.getvalue() + b"\0" * 8)
    else:
        stream = io.BytesIO()
        array = np.full((4, 8), np.nan) if kind == "nan" else np.ones((8, 4)) if kind == "wrong-shape" else np.array([["data"]], dtype=object)
        np.save(stream, array, allow_pickle=True)
        change_member(manifest, members, 0, stream.getvalue())
    monkeypatch.setattr(pickle, "load", lambda *args, **kwargs: pytest.fail("pickle must never be called"))
    monkeypatch.setattr(pickle, "loads", lambda *args, **kwargs: pytest.fail("pickle must never be called"))
    with pytest.raises(BundleError):
        loads_models(repack(manifest, members))


@pytest.mark.parametrize("kind", ["traversal", "undeclared", "compression", "duplicate", "symlink", "prefix", "comment", "truncated", "count", "encrypted-local"])
def test_archive_envelope_rejects_malicious_or_ambiguous_members(kind):
    raw = dumps_models(calibration())
    manifest, members = unpack(raw)
    if kind in ("traversal", "undeclared"):
        members["../escape.npy" if kind == "traversal" else "arrays/a9999.npy"] = b"data"
        raw = repack(manifest, members)
    elif kind == "compression": raw = repack(manifest, members, compression=zipfile.ZIP_DEFLATED)
    elif kind == "duplicate":
        output = io.BytesIO(raw)
        with warnings.catch_warnings(), zipfile.ZipFile(output, "a") as archive:
            warnings.simplefilter("ignore")
            archive.writestr("manifest.json", b"{}")
        raw = output.getvalue()
    elif kind == "symlink":
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            for name, content in [("manifest.json", json.dumps(manifest).encode())] + list(members.items()):
                info = zipfile.ZipInfo(name); info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, content)
        raw = output.getvalue()
    elif kind == "prefix": raw = b"MZ" + raw
    elif kind == "comment":
        output = io.BytesIO(raw)
        with zipfile.ZipFile(output, "a") as archive: archive.comment = b"extra"
        raw = output.getvalue()
    elif kind == "truncated": raw = raw[:-3]
    elif kind == "count":
        raw = bytearray(raw)
        struct.pack_into("<H", raw, len(raw)-12, 65535)
        raw = bytes(raw)
    else:
        raw = bytearray(raw); struct.pack_into("<H", raw, 6, 1); raw = bytes(raw)
    with pytest.raises(BundleError): loads_models(raw)


def test_strict_json_duplicate_keys_and_nonfinite_literals_rejected():
    raw = dumps_models(MyModels())
    manifest, members = unpack(raw)
    for payload in (b'{"format":"a","format":"b"}', b'{"x":NaN}', b'{"x":Infinity}'):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive: archive.writestr("manifest.json", payload)
        with pytest.raises(BundleError): loads_models(output.getvalue())


@pytest.mark.parametrize("field,value", [("max_archive_bytes", 64), ("max_array_bytes", 64),
    ("max_total_array_bytes", 64), ("max_manifest_bytes", 64), ("max_arrays", 1), ("max_dimension", 2)])
def test_custom_resource_bounds_are_enforced(field, value):
    raw = dumps_models(calibration())
    with pytest.raises(BundleError): loads_models(raw, limits=BundleLimits(**{field: value}))


@pytest.mark.parametrize("field,value", [("max_arrays", True), ("max_array_bytes", 0), ("max_arrays", 10000)])
def test_invalid_limits_rejected(field, value):
    with pytest.raises(ValueError): BundleLimits(**{field: value})


@pytest.mark.parametrize("failure", ["fsync", "readback", "replace"])
def test_atomic_failures_preserve_existing_destination(tmp_path, monkeypatch, failure):
    path = tmp_path / "calibration.ccpb"
    original = dumps_models(MyModels())
    path.write_bytes(original)
    def fail(*args, **kwargs): raise OSError("injected")
    if failure == "fsync": monkeypatch.setattr(bundle.os, "fsync", fail)
    elif failure == "readback": monkeypatch.setattr(bundle, "load_bundle", fail)
    else: monkeypatch.setattr(bundle.os, "replace", fail)
    with pytest.raises(OSError): save_bundle(calibration(), path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_unsupported_estimator_has_no_opaque_fallback(tmp_path):
    model = calibration()
    model.model_cc[0].model = object()
    path = tmp_path / "model.ccpb"
    path.write_bytes(b"previous")
    with pytest.raises(BundleError, match="unsupported"):
        save_bundle(model, path)
    assert path.read_bytes() == b"previous"


def test_fresh_process_loading_uses_only_bundle_parameters(tmp_path):
    models = calibration("pls", modern=True, built=True)
    x, _ = paired()
    save_bundle(models, tmp_path / "model.ccpb")
    np.save(tmp_path / "input.npy", x[:4])
    np.save(tmp_path / "expected.npy", predict_(x[:4], models.model_cc[0]))
    code = r'''
import os,sys,importlib.util,numpy as np
from pathlib import Path
import ColorCorrectionPipeline
stage=os.environ.get('CCP_BUNDLE_STAGE')
if stage:
    ColorCorrectionPipeline.__path__.insert(0,stage)
    for short in ('portable_inference','bundle'):
        name='ColorCorrectionPipeline.'+short
        spec=importlib.util.spec_from_file_location(name,Path(stage)/(short+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        setattr(ColorCorrectionPipeline,short,module)
from ColorCorrectionPipeline.bundle import load_bundle
from ColorCorrectionPipeline.core.correction import predict_
root=Path(sys.argv[1]);model=load_bundle(root/'model.ccpb')
np.testing.assert_array_equal(predict_(np.load(root/'input.npy'),model.model_cc[0]),np.load(root/'expected.npy'))
'''
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    source = Path(bundle.__file__).parent
    environment["CCP_BUNDLE_STAGE"] = str(source)
    subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)], env=environment,
                   check=True, capture_output=True, timeout=120)


def test_integer_legacy_identity_parameters_convert_without_loss():
    models = MyModels()
    models.model_ffc = np.ones((4, 8), dtype=np.int16)
    models.model_gc = np.array([1, 0], dtype=np.int64)
    models.model_wb = np.eye(3, dtype=np.int32)
    restored = loads_models(dumps_models(models))
    np.testing.assert_array_equal(restored.model_gc, models.model_gc)
    np.testing.assert_array_equal(restored.model_ffc, models.model_ffc)
    np.testing.assert_array_equal(restored.model_wb, models.model_wb)


def test_large_integer_precision_loss_is_explicitly_rejected():
    models = MyModels()
    models.model_gc = np.array([2**53 + 1, 0], dtype=np.int64)
    with pytest.raises(BundleError, match="exact float64"):
        dumps_models(models)


def test_legacy_constant_regression_and_gamma_remain_supported():
    x, y = paired()
    model = fit_model(x, y, {"degree": 0, "mtd": "linear", "use_lut": False})
    container = MyModels()
    container.model_gc = np.array([20])
    container.model_cc = (model, {}, "ours")
    restored = loads_models(dumps_models(container))
    np.testing.assert_array_equal(predict_(x, restored.model_cc[0]), predict_(x, model))
    np.testing.assert_array_equal(restored.model_gc, [20.])


@pytest.mark.parametrize("method", ["linear", "pls", "nn", "custom", "conv"])
def test_complete_modern_metadata_survives_known_estimator_reconstruction(monkeypatch, method):
    from tests.test_unit.test_modern_pipeline import chart, config
    image, reference = chart(monkeypatch)
    parameters = {"degree": 1}
    if method == "conv": parameters["cc_method"] = "conv"
    else: parameters["mtd"] = method
    if method in ("nn", "custom"):
        parameters.update(max_iterations=3, hidden_layers=[4], batch_size=8, patience=2)
    pipeline = ColorCorrection()
    cfg = config(reference_profile=reference, CC_kwargs=parameters, validation_options={"enabled": False})
    assert not pipeline.run(image, config=cfg)[2], pipeline.last_run
    restored = loads_models(dumps_models(pipeline.models))
    assert restored.metadata == pipeline.models.metadata
    expected = pipeline.predict_image(image)["CC"]
    other = ColorCorrection()
    other.models = restored
    np.testing.assert_allclose(other.predict_image(image)["CC"], expected, atol=2e-12, rtol=2e-12)


def test_degree_resource_bomb_rejected_before_feature_generation(monkeypatch):
    manifest, members = unpack(dumps_models(calibration()))
    manifest["stages"]["CC"]["attributes"]["degree"] = 10**9
    monkeypatch.setattr("sklearn.preprocessing.PolynomialFeatures", lambda *args, **kwargs: pytest.fail("must reject degree before features"))
    with pytest.raises(BundleError, match="degree"):
        loads_models(repack(manifest, members))


def test_repeated_network_array_aliases_count_toward_expanded_parameter_limit():
    models = calibration("nn", modern=True)
    manifest, members = unpack(dumps_models(models))
    estimator = manifest["stages"]["CC"]["estimator"]
    with pytest.raises(BundleError, match="network parameter"):
        loads_models(repack(manifest, members), limits=BundleLimits(max_network_parameters=3))


def test_snapshot_torch_reconstruction_does_not_consume_caller_rng():
    import torch
    model = calibration("custom", modern=True)
    before = torch.random.get_rng_state().clone()
    loads_models(dumps_models(model))
    assert torch.equal(torch.random.get_rng_state(), before)


def test_digest_uses_the_same_stage_shape_gate_as_bundle_loading():
    models = MyModels()
    models.model_wb = np.eye(4)
    with pytest.raises(BundleError):
        digest_models(models)


def test_npy_header_length_resource_bomb_rejected_before_numpy_header_reader(monkeypatch):
    manifest, members = unpack(dumps_models(calibration()))
    raw = b"\x93NUMPY" + bytes([2, 0]) + struct.pack("<I", 2**32 - 1) + b"{}"
    change_member(manifest, members, 0, raw)
    monkeypatch.setattr(np.lib.format, "read_array_header_2_0", lambda *a, **k: pytest.fail("must reject before large header read"))
    with pytest.raises(BundleError, match="header length"):
        loads_models(repack(manifest, members))


def test_disk_loader_rejects_nonregular_path_before_open(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(Path, "stat", lambda *args, **kwargs: SimpleNamespace(st_mode=stat.S_IFIFO, st_size=0))
    monkeypatch.setattr(bundle.os, "open", lambda *args, **kwargs: pytest.fail("must not open nonregular paths"))
    with pytest.raises(BundleError, match="regular file"):
        load_bundle(tmp_path / "pipe")


def test_disk_loader_rechecks_open_descriptor_and_closes_on_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    path = tmp_path / "model.ccpb"
    path.write_bytes(dumps_models(MyModels()))
    original_open, original_fstat = os.open, os.fstat
    descriptors = []
    def observed_open(*args, **kwargs):
        descriptor = original_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor
    monkeypatch.setattr(bundle.os, "open", observed_open)
    monkeypatch.setattr(bundle.os, "fstat", lambda *args: SimpleNamespace(st_mode=stat.S_IFREG, st_size=2**40))
    with pytest.raises(BundleError, match="bounded regular"):
        load_bundle(path)
    with pytest.raises(OSError):
        original_fstat(descriptors[-1])


def test_structured_float_union_header_cannot_masquerade_as_plain_float():
    manifest, members = unpack(dumps_models(calibration()))
    output = io.BytesIO()
    np.lib.format.write_array_header_1_0(output, {"descr": ("<f8", {"x": ("<i8", 0)}),
        "fortran_order": False, "shape": (4, 8)})
    raw = output.getvalue() + np.ones((4, 8), dtype="<f8").tobytes()
    change_member(manifest, members, 0, raw)
    with pytest.raises(BundleError, match="unstructured primitive"):
        loads_models(repack(manifest, members))


@pytest.mark.parametrize("kind", ["root-forward", "root-pre", "child-forward", "instance-forward"])
def test_torch_graph_hooks_and_overrides_are_not_silently_stripped(kind):
    models = calibration("custom", modern=True)
    graph = models.model_cc[0].model.model
    handle = None
    if kind == "root-forward": handle = graph.register_forward_hook(lambda module, args, output: output * 2)
    elif kind == "root-pre": handle = graph.register_forward_pre_hook(lambda module, args: (args[0] * 2,))
    elif kind == "child-forward": handle = graph[0].register_forward_hook(lambda module, args, output: output * 2)
    else: graph.forward = lambda values: values[:, :3]
    try:
        with pytest.raises(BundleError, match="hooks|overrides"):
            dumps_models(models)
    finally:
        if handle is not None: handle.remove()


def test_estimator_predict_override_is_explicitly_unsupported():
    models = calibration()
    models.model_cc[0].model.predict = lambda values: np.zeros((len(values), 3))
    with pytest.raises(BundleError, match="overrides"):
        dumps_models(models)


def test_rejected_candidate_is_not_retained_by_lut_callback_cycle(monkeypatch):
    import gc
    import weakref
    from ColorCorrectionPipeline.portable_inference import AffinePredictor
    raw = dumps_models(calibration(built=True))
    observed = []
    def fail(candidate):
        assert type(candidate.model_cc[0].model) is AffinePredictor
        observed.append(weakref.ref(candidate.model_cc[0]))
        raise ValueError("injected metadata validation failure")
    monkeypatch.setattr(MyModels, "_validated_state", staticmethod(fail))
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        def rejected():
            try:
                loads_models(raw)
            except BundleError:
                return
            pytest.fail("must reject injected metadata failure")
        rejected()
        assert observed and observed[0]() is None
    finally:
        if was_enabled: gc.enable()


def test_scalar_intercept_keeps_its_precision_independent_of_coefficient_dtype():
    models = calibration()
    estimator = models.model_cc[0].model
    estimator.coef_ = estimator.coef_.astype(np.float32)
    estimator.intercept_ = np.float64(.123456789123456)
    x, _ = paired()
    expected = predict_(x[:4], models.model_cc[0])
    restored = loads_models(dumps_models(models))
    np.testing.assert_array_equal(predict_(x[:4], restored.model_cc[0]), expected)
