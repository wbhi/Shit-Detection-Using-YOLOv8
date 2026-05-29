"""
Ship Detection Agent
====================
Detects and classifies ships in optical (RGB) and SAR (radar/grayscale)
satellite images using a two-stage pipeline:

  Stage 1 — Satellite-trained YOLO model (HuggingFace) finds candidates
  Stage 2 — COCO YOLO validator rejects land vehicles (cars, trucks, people)

Run:
    python agent.py

UI available at http://localhost:7860
"""

import os
import warnings

os.environ["MPLBACKEND"] = "Agg"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import gradio as gr

import config


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single ship detection with its bounding box, confidence and label."""
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    label: str

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def scaled(self, factor: float) -> "Detection":
        """Return a new Detection with coordinates scaled by factor."""
        return Detection(
            x1=int(self.x1 * factor), y1=int(self.y1 * factor),
            x2=int(self.x2 * factor), y2=int(self.y2 * factor),
            conf=self.conf, label=self.label,
        )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_hf_model(repo: str, filename: str) -> tuple[YOLO, bool]:
    """Download model from HuggingFace and load it. Returns (model, success)."""
    try:
        path = hf_hub_download(repo_id=repo, filename=filename)
        model = YOLO(path)
        print(f"  ✓ {repo}/{filename}")
        return model, True
    except Exception as exc:
        print(f"  ✗ {repo}/{filename} — {exc}. Falling back to {config.COCO_MODEL_FILE}")
        return YOLO(config.COCO_MODEL_FILE), False


print("Loading models...")
optical_model, USE_HF_OPTICAL = _load_hf_model(config.OPTICAL_MODEL_REPO, config.OPTICAL_MODEL_FILE)
sar_model,     USE_HF_SAR     = _load_hf_model(config.SAR_MODEL_REPO,     config.SAR_MODEL_FILE)
coco_model                    = YOLO(config.COCO_MODEL_FILE)
print("All models ready.\n")


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def preprocess(image: Image.Image) -> np.ndarray:
    """Convert PIL image to BGR numpy array, dropping alpha if present."""
    arr = np.array(image)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def resize_for_inference(img: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Downscale image so its longest side ≤ MAX_INFER_DIM.
    Returns (resized_image, scale_factor). Scale = 1.0 if no resize needed.
    """
    h, w = img.shape[:2]
    scale = min(config.MAX_INFER_DIM / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img, scale


def is_sar(img: np.ndarray) -> bool:
    """
    Return True if the image appears to be SAR (grayscale/radar).
    Checks mean colour channel difference — SAR images have near-zero saturation.
    """
    if img.ndim == 2:
        return True
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    diff = (np.mean(np.abs(r.astype(int) - g.astype(int))) +
            np.mean(np.abs(g.astype(int) - b.astype(int))))
    return diff < config.SAR_CHANNEL_DIFF_THRESHOLD


# ---------------------------------------------------------------------------
# Detection pipeline
# ---------------------------------------------------------------------------

def run_yolo(img: np.ndarray, model: YOLO, conf: float) -> list[Detection]:
    """Run YOLO inference and return raw Detection objects above MIN_BOX_AREA."""
    results = model(img, conf=conf, iou=0.5, verbose=False)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if area < config.MIN_BOX_AREA:
            continue
        cls_id = int(box.cls[0])
        label = model.names.get(cls_id, "") if hasattr(model, "names") else ""
        detections.append(Detection(x1, y1, x2, y2, float(box.conf[0]), label))
    return detections


def filter_by_keyword(detections: list[Detection]) -> list[Detection]:
    """
    Stage 1 filter: keep only detections whose label contains a ship keyword,
    or detections with an empty label (classified by size later).
    """
    return [
        d for d in detections
        if not d.label
        or any(kw in d.label.lower() for kw in config.SHIP_KEYWORDS)
    ]


def validate_with_coco(img: np.ndarray, detections: list[Detection]) -> list[Detection]:
    """
    Stage 2 filter: crop each detection region, run COCO model on the crop.
    - If COCO sees a land vehicle at high confidence → reject the detection.
    - If COCO sees a boat → boost confidence by COCO_CONF_BOOST.
    - Otherwise → keep as-is.
    Only applied to optical images (SAR looks nothing like COCO training data).
    """
    if not detections:
        return []

    h, w = img.shape[:2]
    validated = []

    for det in detections:
        pad_x = max(int(det.width  * config.COCO_CROP_PADDING), 5)
        pad_y = max(int(det.height * config.COCO_CROP_PADDING), 5)
        crop = img[
            max(0, det.y1 - pad_y) : min(h, det.y2 + pad_y),
            max(0, det.x1 - pad_x) : min(w, det.x2 + pad_x),
        ]
        if crop.size == 0:
            validated.append(det)
            continue

        coco_results = coco_model(crop, conf=config.COCO_VALIDATOR_CONF, verbose=False)
        rejected = False
        boat_confirmed = False

        for cb in coco_results[0].boxes:
            cls_id   = int(cb.cls[0])
            coco_conf = float(cb.conf[0])
            if cls_id in config.COCO_NON_SHIP_CLASSES and coco_conf > config.COCO_REJECT_CONF:
                rejected = True
                break
            if cls_id == config.COCO_BOAT_CLASS:
                boat_confirmed = True

        if rejected:
            continue

        if boat_confirmed:
            det = Detection(det.x1, det.y1, det.x2, det.y2,
                            min(det.conf + config.COCO_CONF_BOOST, 1.0), det.label)
        validated.append(det)

    return validated


def apply_nms(detections: list[Detection]) -> list[Detection]:
    """Remove overlapping duplicate boxes using Non-Maximum Suppression."""
    if not detections:
        return []
    coords = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
    scores = np.array([d.conf for d in detections], dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(
        coords.tolist(), scores.tolist(),
        score_threshold=0.0, nms_threshold=config.NMS_IOU_THRESHOLD,
    )
    if len(indices) == 0:
        return []
    return [detections[i] for i in indices.flatten()]


def classify_vessel(det: Detection) -> str:
    """
    Return a human-readable ship type.
    Uses the model's label if it's specific; otherwise infers from box size/shape.
    """
    label = det.label.lower() if det.label else ""
    generic = {"ship", "boat", "vessel", "0", ""}
    if label and label not in generic:
        return det.label.title()

    ratio = max(det.width, det.height) / max(min(det.width, det.height), 1)
    if det.area   > config.VESSEL_LARGE_AREA:      return "Cargo / Tanker"
    if det.area   > config.VESSEL_MEDIUM_AREA:     return "Container Ship"
    if ratio      > config.VESSEL_ELONGATED_RATIO: return "Bulk Carrier"
    if det.area   < config.VESSEL_SMALL_AREA:      return "Fishing Vessel"
    return "Vessel"


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate(img: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bounding boxes, numbered circles and labels onto img (in-place)."""
    h, w = img.shape[:2]

    for idx, det in enumerate(detections):
        color      = config.BOX_COLORS[idx % len(config.BOX_COLORS)]
        ship_type  = classify_vessel(det)
        label_text = f"{ship_type} {det.conf:.0%}"

        # Bounding box
        cv2.rectangle(img, (det.x1, det.y1), (det.x2, det.y2), color, 3)

        # Numbered circle (clamped inside image)
        cx = max(15, min((det.x1 + det.x2) // 2, w - 15))
        cy = max(18, min(det.y1 - 18, h - 15))
        cv2.circle(img, (cx, cy), 15, color, -1)
        cv2.putText(img, str(idx + 1), (cx - 6, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Label (clamped so it never goes above y=0)
        (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y_top = max(det.y1 - lh - 32, 0)
        label_y_bot = max(det.y1 - 28, lh + 4)
        cv2.rectangle(img, (det.x1, label_y_top), (det.x1 + lw + 6, label_y_bot), color, -1)
        cv2.putText(img, label_text, (det.x1 + 3, label_y_bot - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    return img


def draw_summary_banner(img: np.ndarray, image_type: str, count: int) -> np.ndarray:
    """Draw a top-left summary banner showing image type and ship count."""
    text = f"[{image_type}]  Ships Found: {count}"
    (sw, sh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(img, (5, 5), (sw + 18, sh + 16), (0, 0, 0), -1)
    cv2.putText(img, text, (10, sh + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return img


# ---------------------------------------------------------------------------
# Main detection function (called by Gradio)
# ---------------------------------------------------------------------------

def detect_ships(
    image: Optional[Image.Image],
    conf_threshold: float = config.DEFAULT_CONF_THRESHOLD,
    image_type: str = "Auto-detect",
) -> tuple[Optional[Image.Image], str]:
    """
    Full detection pipeline:
      preprocess → resize → detect → keyword filter → COCO validate → NMS → annotate
    """
    if image is None:
        return None, "No image provided."

    img_bgr = preprocess(image)

    # Resolve image type
    detected_type = (
        ("SAR" if is_sar(img_bgr) else "Optical")
        if image_type == "Auto-detect"
        else image_type
    )

    # Select model
    if detected_type == "SAR":
        active_model = sar_model
        hf_used      = USE_HF_SAR
    else:
        active_model = optical_model
        hf_used      = USE_HF_OPTICAL

    # Resize for inference (scale back coordinates afterwards)
    img_infer, scale = resize_for_inference(img_bgr)

    # Stage 1a — YOLO inference
    candidates = run_yolo(img_infer, active_model, conf_threshold)

    # Stage 1b — keyword label filter
    candidates = filter_by_keyword(candidates)

    # Stage 2 — COCO validator (optical only)
    if detected_type == "Optical":
        candidates = validate_with_coco(img_infer, candidates)

    # Scale boxes back to original resolution
    if scale < 1.0:
        candidates = [d.scaled(1.0 / scale) for d in candidates]

    # Remove duplicate overlapping boxes
    final = apply_nms(candidates)

    # Annotate original image
    annotate(img_bgr, final)
    draw_summary_banner(img_bgr, detected_type, len(final))

    result_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # Build text summary
    model_label = (
        "Satellite (HuggingFace) + COCO validator" if hf_used and detected_type == "Optical"
        else "SAR model (HuggingFace)"              if hf_used
        else "YOLOv8 fallback (COCO)"
    )

    if not final:
        summary = (
            f"Image type: {detected_type}\n"
            f"Model: {model_label}\n\n"
            "No ships found.\n\n"
            "Tips:\n"
            "  • Lower the confidence threshold slider\n"
            "  • Try switching image type manually\n"
            "  • For SAR images, ensure image is grayscale"
        )
    else:
        lines = [f"  {i+1}. {classify_vessel(d)} ({d.conf:.0%})" for i, d in enumerate(final)]
        summary = (
            f"Image type: {detected_type}\n"
            f"Model: {model_label}\n\n"
            f"Detected {len(final)} ship(s):\n" + "\n".join(lines)
        )

    return result_image, summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

demo = gr.Interface(
    fn=detect_ships,
    inputs=[
        gr.Image(type="pil", label="Upload Satellite Image (Optical or SAR)"),
        gr.Slider(
            minimum=0.05, maximum=0.9,
            value=config.DEFAULT_CONF_THRESHOLD, step=0.05,
            label="Confidence Threshold  (≥0.35 recommended)",
        ),
        gr.Radio(
            choices=["Auto-detect", "Optical", "SAR"],
            value="Auto-detect",
            label="Image Type",
        ),
    ],
    outputs=[
        gr.Image(type="pil", label="Annotated Image"),
        gr.Textbox(label="Detection Summary", lines=10),
    ],
    title="Ship Detection Agent",
    description=(
        "Upload an optical (RGB) or SAR (radar/grayscale) satellite image.\n\n"
        "Two-stage detection pipeline:\n"
        "  Stage 1 — Satellite model finds ship candidates\n"
        "  Stage 2 — COCO validator rejects cars, trucks, people\n\n"
        "Confidence threshold guide:\n"
        "  ≥ 0.35 — recommended, clean results\n"
        "  0.20–0.34 — use when ships are missed; COCO validator still filters cars\n"
        "  < 0.20 — very sensitive; some false positives may appear"
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(share=False)
