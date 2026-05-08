import numpy as np

from ColorCorrectionPipeline.constants import MODEL_PATH
from ColorCorrectionPipeline.flat_field import FlatFieldCorrection


def test_model_path_uses_onnx_detector():
    assert MODEL_PATH.endswith(".onnx")


def test_opencv_dnn_output_parser_maps_box_to_original_image():
    ffc = FlatFieldCorrection(
        np.zeros((512, 512, 3), dtype=np.uint8),
        manual_crop=True,
        conf_threshold=0.7,
    )
    output = np.zeros((1, 5, 10), dtype=np.float32)
    output[0, :, 0] = [256, 256, 100, 80, 0.9]
    output[0, :, 1] = [256, 256, 100, 80, 0.1]

    boxes, scores = ffc._parse_yolo_output(
        output,
        scale=1.0,
        pad_x=0,
        pad_y=0,
        img_shape=ffc.img.shape,
    )

    assert boxes.tolist() == [[206, 216, 306, 296]]
    assert np.allclose(scores, [0.9])


def test_crop_rect_skips_model_and_clamps_to_image_bounds():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    ffc = FlatFieldCorrection(
        img,
        model_path="missing-detector.onnx",
        crop_rect=(-10, 5, 250, 120),
    )

    ffc.detect_and_crop()

    assert ffc.model is None
    assert ffc.crop_rect == [0, 5, 200, 100]
    assert ffc.img_cropped.shape == (95, 200, 3)


def test_polynomial_features_respect_interactions_flag():
    ffc = FlatFieldCorrection(np.zeros((10, 10, 3), dtype=np.uint8), manual_crop=True)
    X = np.array([[2.0, 3.0]])

    powers_only, names, _ = ffc.polynomial_features(X, degree=2, interactions=False)
    with_interactions, interaction_names, _ = ffc.polynomial_features(
        X,
        degree=2,
        interactions=True,
    )

    assert names.tolist() == ["1", "x", "y", "x^2", "y^2"]
    assert powers_only.tolist() == [[1.0, 2.0, 3.0, 4.0, 9.0]]
    assert "x y" in interaction_names.tolist()
    assert with_interactions.shape[1] > powers_only.shape[1]
