"""
Flat-field white-plane detection and safe-crop validation
=========================================================

``FlatFieldCorrection`` finds the white reference plane in a white-field image
(bundled ONNX detector, an explicit ``crop_rect`` or manual ROI selection) and
enforces the safe-mode signal, saturation, coverage and gain policies. The
pipeline's FFC fitter (``ui_stages.fit_ffc``) builds the gain surface from that
crop. 3.0 removed the 2.x multiplier fitters (``compute_multiplier`` with its
linear/NN/PLS/SVM backends), their plotting helpers and ``apply_ffc``; saved FFC
gains still apply through the pipeline.

Example:
    >>> import cv2
    >>> from ColorCorrectionPipeline.flat_field import FlatFieldCorrection
    >>>
    >>> white_img = cv2.imread("white_background.jpg")
    >>> ffc = FlatFieldCorrection(white_img, bins=50, smooth_window=5, safe_mode=True)
    >>> ffc.detect_and_crop()
    >>> x1, y1, x2, y2 = ffc.crop_rect
"""

import gc
import logging
import os
import warnings
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from ..constants import MODEL_PATH

logger = logging.getLogger(__name__)

__all__ = ["FlatFieldCorrection"]

UINT8 = np.uint8


class FlatFieldCorrection:
    """
    White-plane detection and safe crop validation for flat-field correction.

    Attributes:
        img: Input BGR image (uint8)
        model_path: Path to OpenCV DNN ONNX model for plane detection
        manual_crop: Whether to use manual ROI selection
        show: Whether to display intermediate plots
        bins: Number of bins for intensity sampling
        smooth_window: Window size for Gaussian smoothing
        crop_rect: Manual crop rectangle [x1, y1, x2, y2]
        model: OpenCV DNN model instance (if available)
        img_cropped: Cropped white plane region
        is_color: Whether image is color (vs grayscale)

    Example:
        >>> ffc = FlatFieldCorrection(img=white_img, bins=50, smooth_window=5, safe_mode=True)
        >>> ffc.detect_and_crop()
    """

    def __init__(self, img: Optional[np.ndarray] = None, **kwargs):
        """
        Initialize FlatFieldCorrection.

        Args:
            img: BGR image (uint8), typically white background image
            **kwargs: Configuration parameters
                model_path: Path to ONNX detection model (default: MODEL_PATH constant)
                manual_crop: Force manual ROI selection (default: False)
                show: Display intermediate plots (default: False)
                bins: Bins for intensity sampling (default: 50)
                smooth_window: Gaussian smoothing window size (default: 5)
                crop_rect: Pre-defined crop rectangle [x1, y1, x2, y2]
        """
        self.img = img
        # Detection uses a byte preview. Precision-aware callers can supply
        # original RGB samples for independent signal and saturation checks.
        self.quality_source = kwargs.get("quality_source")
        if self.quality_source is not None:
            source = np.array(self.quality_source, dtype=np.float64, copy=True)
            if (
                img is None
                or source.shape != (*img.shape[:2], 3)
                or not np.isfinite(source).all()
                or np.any(source < 0)
                or np.any(source > 1)
            ):
                raise ValueError("FFC quality_source must match the preview as finite RGB in [0,1]")
            source.setflags(write=False)
            self.quality_source = source
        self.model_path = kwargs.get("model_path", MODEL_PATH)
        self.manual_crop = kwargs.get("manual_crop", False)
        self.show = kwargs.get("show", False)
        self.bins = kwargs.get("bins", 50)
        self.smooth_window = kwargs.get("smooth_window", 5)
        self.crop_rect = kwargs.get("crop_rect", None)
        self.conf_threshold = kwargs.get("conf_threshold", 0.7)
        self.iou_threshold = kwargs.get("iou_threshold", 0.6)
        self.dnn_input_size = kwargs.get("dnn_input_size", 512)
        self.safe_mode = kwargs.get("safe_mode", False)
        self.min_signal = kwargs.get("min_signal", 0.02)
        self.max_saturated_fraction = kwargs.get("max_saturated_fraction", 0.01)
        self.min_crop_fraction = kwargs.get("min_crop_fraction", 0.1)
        self.min_gain = kwargs.get("min_gain", 0.5)
        self.max_gain = kwargs.get("max_gain", 1.8)
        self.missing_plane = kwargs.get("missing_plane", "error")
        self.quality_report = {
            "algorithm": "legacy-uint8-Lab-lightness",
            "safe_mode": self.safe_mode,
        }
        if not isinstance(self.safe_mode, bool):
            raise TypeError("FFC safe_mode must be boolean")
        if self.safe_mode:
            self._validate_safe_configuration()

        self.model: Optional[Any] = None
        self.img_cropped: Optional[np.ndarray] = None
        self.is_color = self.check_color(self.img) if self.img is not None else None

        if not self.manual_crop and self.crop_rect is None:
            self.model = self._load_detection_model()

    def _validate_safe_configuration(self):
        if self.img is None or not isinstance(self.img, np.ndarray) or self.img.dtype != UINT8:
            raise ValueError(
                "safe FFC requires a uint8 white image; higher precision radiometric FFC is not implemented"
            )
        if (
            self.img.ndim not in (2, 3)
            or (self.img.ndim == 3 and self.img.shape[2] != 3)
            or min(self.img.shape[:2]) < 4
        ):
            raise ValueError(
                "safe FFC requires a nonempty grayscale or BGR white image with both dimensions >= 4"
            )
        for name in ("bins", "smooth_window"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < (2 if name == "bins" else 1)
            ):
                raise ValueError("safe FFC %s must be a positive integer" % name)
        if self.smooth_window % 2 != 1:
            raise ValueError("safe FFC smooth_window must be odd")
        for name in (
            "min_signal",
            "max_saturated_fraction",
            "min_crop_fraction",
            "min_gain",
            "max_gain",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int, np.number))
                or not np.isfinite(value)
            ):
                raise ValueError("safe FFC %s must be finite numeric" % name)
        if (
            not 0 < self.min_signal < 1
            or not 0 <= self.max_saturated_fraction <= 1
            or not 0 < self.min_crop_fraction <= 1
        ):
            raise ValueError(
                "safe FFC signal, saturation and coverage policies are outside their supported ranges"
            )
        if not 0 < self.min_gain <= self.max_gain or self.max_gain > 100:
            raise ValueError("safe FFC requires 0 < min_gain <= max_gain <= 100")
        if self.missing_plane not in ("error", "full-image"):
            raise ValueError("safe FFC missing_plane must be error or full-image")
        if self.manual_crop and self.crop_rect is None:
            raise ValueError(
                "safe FFC manual_crop requires an explicit crop_rect; interactive selection is not reproducible"
            )
        if self.crop_rect is not None:
            self._normalize_crop_rect(tuple(self.crop_rect))

    def _validate_safe_crop(self):
        height, width = self.img.shape[:2]
        crop_height, crop_width = self.img_cropped.shape[:2]
        fraction = crop_height * crop_width / (height * width)
        if fraction < self.min_crop_fraction:
            raise ValueError("safe FFC crop has insufficient spatial coverage")
        if self.bins > min(crop_height, crop_width):
            raise ValueError("safe FFC bins exceed distinct crop pixel coordinates")
        if self.smooth_window > min(crop_height, crop_width):
            raise ValueError("safe FFC smoothing window exceeds crop dimensions")
        if self.quality_source is None:
            raw = self.img_cropped
            clipped = np.any(raw >= 253, axis=2) if raw.ndim == 3 else raw >= 253
            raw_lightness, _ = self.get_L(raw, smooth=False)
            minimum_lightness = float(raw_lightness.min()) / 255
        else:
            from ..ui_native import lab

            x1, y1, x2, y2 = self.crop_rect
            raw = self.quality_source[y1:y2, x1:x2]
            clipped = np.any(raw >= 253.0 / 255, axis=2)
            minimum_lightness = float(lab(raw)[..., 0].min()) / 100
        saturated_fraction = float(np.mean(clipped))
        self.quality_report.update(
            {
                "crop_rect": list(self.crop_rect),
                "crop_fraction": fraction,
                "source_shape": list(self.img.shape[:2]),
                "saturated_pixel_fraction": saturated_fraction,
                "saturation_threshold_uint8": 253,
                "minimum_raw_normalized_lightness": minimum_lightness,
                "min_signal": float(self.min_signal),
                "gain_limits": [float(self.min_gain), float(self.max_gain)],
            }
        )
        if self.quality_source is not None:
            self.quality_report.update(
                {
                    "measurement_source": "original encoded RGB; float64",
                    "saturation_threshold_encoded": 253.0 / 255,
                }
            )
        if saturated_fraction > self.max_saturated_fraction:
            raise ValueError("safe FFC white plane is saturated beyond max_saturated_fraction")
        if minimum_lightness < self.min_signal:
            raise ValueError("safe FFC white plane has insufficient signal before smoothing")

    def check_color(self, img: np.ndarray) -> bool:
        """Check if image is color (3 channels) vs grayscale."""
        self.is_color = img.ndim == 3 and img.shape[2] == 3
        return self.is_color

    def _load_detection_model(self) -> Optional[Any]:
        """Load the ONNX white-plane detector with OpenCV DNN."""
        if self.model_path == "" or not os.path.exists(self.model_path):
            warnings.warn(
                f"plane detection model not found at {self.model_path}. "
                "Using full image for flat-field correction.",
                stacklevel=2,
            )
            return None

        try:
            return cv2.dnn.readNetFromONNX(self.model_path)
        except cv2.error as exc:
            warnings.warn(
                f"could not load plane detection model at {self.model_path}: {exc}. "
                "Using full image for flat-field correction.",
                stacklevel=2,
            )
            return None

    def _letterbox(self, img: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """Resize image into the model square while preserving aspect ratio."""
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        height, width = img.shape[:2]
        scale = min(self.dnn_input_size / width, self.dnn_input_size / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        pad_x = (self.dnn_input_size - new_width) // 2
        pad_y = (self.dnn_input_size - new_height) // 2
        right = self.dnn_input_size - new_width - pad_x
        bottom = self.dnn_input_size - new_height - pad_y
        padded = cv2.copyMakeBorder(
            resized,
            pad_y,
            bottom,
            pad_x,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return padded, scale, pad_x, pad_y

    def _parse_yolo_output(
        self,
        output: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        img_shape: Tuple[int, ...],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parse YOLO-style ONNX output into original-image xyxy boxes."""
        predictions = np.squeeze(output)
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)
        if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] <= 20:
            predictions = predictions.T

        height, width = img_shape[:2]
        boxes_xywh = []
        boxes_xyxy = []
        confidences = []

        for row in predictions:
            if row.shape[0] < 5:
                continue

            score = float(np.max(row[4:]))
            if score < self.conf_threshold:
                continue

            cx, cy, box_w, box_h = row[:4].astype(float)
            x1 = (cx - box_w / 2 - pad_x) / scale
            y1 = (cy - box_h / 2 - pad_y) / scale
            x2 = (cx + box_w / 2 - pad_x) / scale
            y2 = (cy + box_h / 2 - pad_y) / scale

            x1 = int(np.clip(round(x1), 0, width - 1))
            y1 = int(np.clip(round(y1), 0, height - 1))
            x2 = int(np.clip(round(x2), 0, width))
            y2 = int(np.clip(round(y2), 0, height))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
            boxes_xyxy.append([x1, y1, x2, y2])
            confidences.append(score)

        if not boxes_xyxy:
            return np.empty((0, 4), dtype=int), np.empty((0,), dtype=float)

        keep = cv2.dnn.NMSBoxes(
            boxes_xywh,
            confidences,
            self.conf_threshold,
            self.iou_threshold,
        )
        if len(keep) == 0:
            return np.empty((0, 4), dtype=int), np.empty((0,), dtype=float)

        keep_indices = np.asarray(keep).reshape(-1)
        return (
            np.asarray(boxes_xyxy, dtype=int)[keep_indices],
            np.asarray(confidences, dtype=float)[keep_indices],
        )

    def _detect_plane_opencv_dnn(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run plane detection through OpenCV DNN."""
        if self.model is None:
            return np.empty((0, 4), dtype=int), np.empty((0,), dtype=float)

        model_img, scale, pad_x, pad_y = self._letterbox(img)
        blob = cv2.dnn.blobFromImage(
            model_img,
            scalefactor=1 / 255.0,
            size=(self.dnn_input_size, self.dnn_input_size),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.model.setInput(blob)
        output = self.model.forward()
        return self._parse_yolo_output(output, scale, pad_x, pad_y, img.shape)

    def _normalize_crop_rect(self, rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Clamp a crop rectangle to the image bounds."""
        height, width = self.img.shape[:2]
        if self.safe_mode:
            if len(rect) != 4 or any(
                isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in rect
            ):
                raise ValueError("safe FFC crop_rect requires four integer coordinates")
            x1, y1, x2, y2 = rect
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(
                    "safe FFC crop_rect must be nonempty and wholly inside the source image"
                )
            return tuple(int(v) for v in rect)
        x1, y1, x2, y2 = [int(v) for v in rect]
        x1 = int(np.clip(x1, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        x2 = int(np.clip(x2, x1 + 1, width))
        y2 = int(np.clip(y2, y1 + 1, height))
        return x1, y1, x2, y2

    def detect_and_crop(self):
        """
        Detect white plane using OpenCV DNN or manual ROI selection.

        Sets self.crop_rect and self.img_cropped.
        """
        if self.img is None:
            raise ValueError("Image is required for flat-field correction.")

        if self.crop_rect is not None:
            x1, y1, x2, y2 = self._normalize_crop_rect(tuple(self.crop_rect))
            if self.safe_mode:
                self.quality_report["plane_selection"] = "explicit-crop"
        elif not self.manual_crop:
            # OpenCV DNN detection
            sr = 0.95  # Shrink ratio to avoid edges
            boxes, probs = self._detect_plane_opencv_dnn(self.img)

            if len(boxes) == 0:
                if self.safe_mode and self.missing_plane == "error":
                    raise ValueError(
                        "safe FFC no white plane detected; provide crop_rect or explicitly allow missing_plane='full-image'"
                    )
                if self.safe_mode:
                    self.quality_report["plane_selection"] = "explicit-full-image-fallback"
                warnings.warn("no white plane detected, using full image", stacklevel=2)
                x1 = 0
                y1 = 0
                x2 = self.img.shape[1]
                y2 = self.img.shape[0]
            else:
                if len(boxes) > 1:
                    logger.warning("%d objects detected, using largest", len(boxes))
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    max_index = np.argmax(areas)
                    selected_box = boxes[max_index]
                    selected_prob = probs[max_index]
                    logger.info(
                        "Selected object %d: BB=%s, prob=%.3f",
                        max_index,
                        selected_box,
                        selected_prob,
                    )
                else:
                    selected_box = boxes[0]
                    selected_prob = probs[0]
                    logger.info(
                        "Detected white plane: BB=%s, prob=%.3f", selected_box, selected_prob
                    )

                if self.safe_mode:
                    self.quality_report.update(
                        {
                            "plane_selection": "detected-largest",
                            "detection_count": len(boxes),
                            "selected_confidence": float(selected_prob),
                        }
                    )

                x1, y1, x2, y2 = selected_box
                margin_x = int(round((1 - sr) * (x2 - x1) / 2))
                margin_y = int(round((1 - sr) * (y2 - y1) / 2))
                x1 = int(x1 + margin_x)
                y1 = int(y1 + margin_y)
                x2 = int(x2 - margin_x)
                y2 = int(y2 - margin_y)
                x1, y1, x2, y2 = self._normalize_crop_rect((x1, y1, x2, y2))

        else:
            # Manual ROI selection
            print('Select ROI manually. Press "ENTER" when done selecting ROI')
            cv2.namedWindow("Press 'ENTER' when done", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Press 'ENTER' when done", 1200, 800)
            rect = cv2.selectROI("Press 'ENTER' when done", self.img, True)
            cv2.destroyAllWindows()

            try:
                x1 = int(rect[0])
                y1 = int(rect[1])
                x2 = int(rect[0] + rect[2])
                y2 = int(rect[1] + rect[3])
            except Exception:
                warnings.warn("ROI not selected, using full image", stacklevel=2)
                x1 = 0
                y1 = 0
                x2 = self.img.shape[1]
                y2 = self.img.shape[0]

        self.crop_rect = [x1, y1, x2, y2]
        self.img_cropped = self.img[y1:y2, x1:x2]
        if self.safe_mode:
            self._validate_safe_crop()

        if self.show:
            img = self.img.copy()
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 10)
            cv2.namedWindow("Image ROI", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image ROI", 1200, 800)
            cv2.imshow("Image ROI", img)
            cv2.waitKey(500)
            cv2.destroyAllWindows()

        gc.collect()

    def get_L(
        self, img: np.ndarray, smooth: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract L channel from image.

        Args:
            img: BGR image (uint8)
            smooth: Whether to apply Gaussian smoothing

        Returns:
            tuple: (L_channel, LAB_image)
                - L_channel: Luminance channel (uint8)
                - LAB_image: Full LAB image if color, else None
        """
        is_color = self.check_color(img)
        img_LAB = None

        if is_color:
            img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            L = img_LAB[:, :, 0]
        else:
            L = img

        if smooth:
            L = cv2.GaussianBlur(L, (self.smooth_window, self.smooth_window), 0)

        return L, img_LAB
