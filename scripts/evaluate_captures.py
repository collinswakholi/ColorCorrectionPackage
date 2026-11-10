"""Evaluate fixed calibrations on a session held out from fitting (see docs/capture-study.md)."""
import argparse
import hashlib
import json
from pathlib import Path

import colour
import numpy as np

from ColorCorrectionPipeline import ColorCorrection, Config, ReferenceProfile
from ColorCorrectionPipeline.detection import DetectionOptions, bind_detection_source, detection_context
from ColorCorrectionPipeline.io import read_image_data
from ColorCorrectionPipeline.observations import capture_observations


METHODS = {
    "CCM": {"model": "3x3"}, "RPCC2": {"model": "rpcc2"}, "RPCC3": {"model": "rpcc3"},
    "NN100": {"model": "nn", "degree": 2, "basis": "polynomial"},
}
NN = dict(hidden_layers=[100], max_iterations=10000, patience=1500, learning_rate=.001,
          tol=1e-6, batch_size=24, random_state=42, dropout_rate=0., use_batch_norm=False,
          optim_type="Adam")
CHART = dict(device="cpu", cpu_threads=8, sampling_policy="ui-native-v1",
             samples_per_axis=9, sampling_inset=.75)
D65 = np.array([.31271, .32902])
D50 = np.array([.3457, .3585])


def load_study(manifest):
    spec = json.loads(manifest.read_text(encoding="utf8"))
    if spec.get("precision", "native-byte") not in ("native-byte", "high"):
        raise ValueError("select native-byte or high precision")
    records = [spec["fit"], *spec["held_out"]]
    if len(records) < 2:
        raise ValueError("at least one held-out capture is required")
    for row in records:
        for key in ("capture_id", "session", "image"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError("each capture requires " + key)
        if any(row[key] != row[key].strip() for key in ("capture_id", "session")):
            raise ValueError("capture_id and session must not have surrounding whitespace")
        index = row.get("chart_index", 0)
        if type(index) is not int or index < 0:
            raise ValueError("chart_index must be a nonnegative integer")
    if len({r["capture_id"] for r in records}) != len(records):
        raise ValueError("each exposure requires a unique capture_id")
    if records[0]["session"] in {r["session"] for r in records[1:]}:
        raise ValueError("the fit session cannot also be held out")
    if spec["scope"] not in ("exploratory", "controlled"):
        raise ValueError("scope must be exploratory or controlled")
    if spec["scope"] == "controlled":
        for key in ("physical_chart_id", "controlled_setup", "measurement_record"):
            if not isinstance(spec.get(key), str) or not spec[key].strip():
                raise ValueError("controlled study requires " + key)
        if not (manifest.parent / spec["measurement_record"]).is_file():
            raise ValueError("missing target measurement record")
    ref = ReferenceProfile.from_dict(json.loads((manifest.parent / spec["reference"]).read_text(encoding="utf8")))
    if ref.template_id != "classic_4x6" or not ref.neutral_indices:
        raise ValueError("a Classic24 reference with declared neutral patches is required")
    return spec, ref


def geometry(data, ref, index):
    rgb = data.to_srgb()
    with detection_context(DetectionOptions(**CHART, chart_index=index)):
        bind_detection_source(np.rint(rgb[:, :, ::-1] * 255).astype(np.uint8))
        obs = capture_observations(rgb, data, ref, "source", explicit=True, force=True)
        if index >= len(obs.observations):
            raise ValueError(obs.unavailable_reason or "requested chart face is missing")
        return obs.observations[index].quadrilaterals


def score(rgb, quads, ref):
    """Sample encoded sRGB in float64 independently of the fit and inference code."""
    quads = np.asarray(quads, dtype=np.float64)
    h, w = rgb.shape[:2]
    # 9 x 9 grid over the central 70% of each detector-supplied patch.
    u, v = np.meshgrid(np.linspace(.15, .85, 9), np.linspace(.15, .85, 9))
    u, v = u.ravel()[:, None], v.ravel()[:, None]
    tl, tr, br, bl = np.swapaxes(quads, 0, 1)[:, :, None, :]
    xy = (1-v)*((1-u)*tl + u*tr) + v*((1-u)*bl + u*br)
    x, y = np.clip(xy[..., 0], 0, w-1), np.clip(xy[..., 1], 0, h-1)
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    dx, dy = (x-x0)[..., None], (y-y0)[..., None]
    samples = ((1-dy)*((1-dx)*rgb[y0,x0] + dx*rgb[y0,x1])
               + dy*((1-dx)*rgb[y1,x0] + dx*rgb[y1,x1]))
    with colour.utilities.domain_range_scale("reference"):
        matrix = colour.normalised_primary_matrix([[.64,.33],[.30,.60],[.15,.06]], D65)
        xyz = colour.models.eotf_sRGB(samples.mean(axis=1)) @ matrix.T
        lab = colour.XYZ_to_Lab(xyz, D65)
        reference_xyz = colour.Lab_to_XYZ(ref.canonical_lab_d50, D50)
        adapted = colour.adaptation.chromatic_adaptation_VonKries(
            reference_xyz, colour.xy_to_XYZ(D50), colour.xy_to_XYZ(D65), transform=ref.adaptation)
        errors = colour.delta_E(lab, colour.XYZ_to_Lab(adapted, D65), method="CIE 2000")
    if not np.isfinite(errors).all():
        raise ValueError("nonfinite chart score")
    return {"mean": float(errors.mean()), "median": float(np.median(errors)),
            "p95": float(np.percentile(errors, 95)), "worst": float(errors.max()),
            "worst_cell": ref.cell_ids[int(errors.argmax())],
            "neutral_mean": float(errors[list(ref.neutral_indices)].mean()),
            "clipped_sample_fraction": float(np.any((samples <= 0) | (samples >= 1), axis=-1).mean()),
            "patch_de00": dict(zip(ref.cell_ids, errors.tolist()))}


def evaluate(manifest, output):
    spec, ref = load_study(manifest)
    root = manifest.parent
    train = spec["fit"]
    chart = read_image_data(root / train["image"], frame_index=0)
    white = read_image_data(root / train["white"], frame_index=0) if train.get("white") else None
    seen = {chart.pixel_digest}
    if white is not None:
        if chart.pixel_digest == white.pixel_digest:
            raise ValueError("fit chart reused as the white field")
        seen.add(white.pixel_digest)
    held = []
    for row in spec["held_out"]:
        item = dict(capture_id=row["capture_id"], session=row["session"])
        held.append(item)
        try:
            path = root / row["image"]
            data = read_image_data(path, frame_index=0)
        except Exception as exc:
            item["error"] = str(exc)
            continue
        if row.get("original_high_depth") and data.source_dtype == "uint8":
            raise ValueError("declared original high-depth capture decodes to uint8")
        if data.pixel_digest in seen:
            raise ValueError("duplicate decoded pixels: " + str(path))
        seen.add(data.pixel_digest)
        item.update(pixel_digest=data.pixel_digest, source_dtype=data.source_dtype)
        try:
            quads = geometry(data, ref, row.get("chart_index", 0))
            item.update(quads=quads.tolist(), raw=score(data.to_srgb(), quads, ref))
        except Exception as exc:
            item["error"] = str(exc)

    output.mkdir(parents=True, exist_ok=False)
    precision = spec.get("precision", "native-byte")
    report = {"manifest": spec, "reference_digest": ref.digest, "fit_pixel_digest": chart.pixel_digest,
              "white_pixel_digest": white.pixel_digest if white is not None else None,
              "fit": {}, "held_out": held, "held_out_summary": {}, "models": {}}
    if spec.get("measurement_record"):
        report["measurement_sha256"] = hashlib.sha256((root / spec["measurement_record"]).read_bytes()).hexdigest()
    fit_quads = None
    for label, recipe in METHODS.items():
        solver = dict(use_lut=precision == "native-byte", lut_grid_size=33, lut_min_pixels=1)
        if label == "NN100":
            solver.update(NN)
        config = Config(**recipe, precision=precision, CC_kwargs=solver, reference_profile=ref,
                        do_ffc=white is not None, do_gc=True, do_wb=True, check_saturation=False,
                        image_options={"frame_index": 0}, validation_options={"enabled": False},
                        chart_detection=dict(CHART, chart_index=train.get("chart_index", 0)))
        pipe = ColorCorrection()
        try:
            result = pipe.fit_or_raise(chart, white, "fit", config)
        except Exception as exc:
            report["fit"][label] = {"error": str(exc)}
            report["held_out_summary"][label] = {"expected": len(held), "scored": 0, "reason": "fit failed"}
            continue
        face = pipe.observations.observations[train.get("chart_index", 0)]
        if fit_quads is None:
            fit_quads = face.quadrilaterals
            report["fit_quads"] = fit_quads.tolist()
        elif not np.array_equal(fit_quads, face.quadrilaterals):
            raise ValueError("fit geometry differs across methods")
        pipe.save_bundle(output / (label + ".ccpb"))
        report["models"][label] = {"config": config.to_grouped_dict(),
            "runtime": pipe.models.metadata["fitting_policy"]["ui_runtime"],
            "detector_execution": pipe.models.metadata.get("detector_execution", []),
            "epochs": pipe.models.model_cc[0].fit_report.get("details", {}).get("epochs")}
        report["fit"][label] = {"reported_cc_de00": result.metrics["fit_CC"]["delta_e00_after_mean"],
            "raw": score(chart.to_srgb(), fit_quads, ref),
            "pre_cc": score(result.images["fit_WB"], fit_quads, ref),
            "cc": score(result.images["fit_CC"], fit_quads, ref)}
        scores, changes = [], []
        for row, item in zip(spec["held_out"], held):
            if "error" in item:
                continue
            try:
                data = read_image_data(root / row["image"], frame_index=0)
                if data.pixel_digest != item["pixel_digest"]:
                    raise ValueError("capture changed during evaluation")
                images = pipe.predict_image(data, stages=("WB", "CC"))
                pre = score(images["WB"], item["quads"], ref)
                cc = score(images["CC"], item["quads"], ref)
                change = cc["mean"] - pre["mean"]
                item.setdefault("methods", {})[label] = {"pre_cc": pre, "cc": cc, "cc_minus_pre_cc": change}
                scores.append(cc["mean"])
                changes.append(change)
            except Exception as exc:
                item.setdefault("methods", {})[label] = {"error": str(exc)}
        summary = {"expected": len(held), "scored": len(scores)}
        if len(scores) == len(held):
            summary.update(mean_de00=float(np.mean(scores)), capture_range=[min(scores), max(scores)],
                           mean_cc_minus_pre_cc=float(np.mean(changes)))
        report["held_out_summary"][label] = summary
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    return all(s["scored"] == s["expected"] for s in report["held_out_summary"].values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path, help="new results directory")
    args = parser.parse_args()
    raise SystemExit(0 if evaluate(args.manifest, args.output) else 1)
