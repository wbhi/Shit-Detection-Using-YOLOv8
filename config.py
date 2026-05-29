"""
Central configuration for the Ship Detection Agent.
All tunable parameters live here — no magic numbers in agent.py.
"""

# ---------------------------------------------------------------------------
# Model sources
# ---------------------------------------------------------------------------
OPTICAL_MODEL_REPO     = "Mahadih534/yolov8_ship_det_satellite"
OPTICAL_MODEL_FILE     = "ship.pt"

SAR_MODEL_REPO         = "hewitleo/sar-ship-detection-yolov8"
SAR_MODEL_FILE         = "weights_(model)/best.pt"

COCO_MODEL_FILE        = "yolov8n.pt"   # local fallback + COCO validator

# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------
MAX_INFER_DIM          = 1280    # max image side before downscaling
MIN_BOX_AREA           = 200     # px² — smaller boxes are noise/wakes
DEFAULT_CONF_THRESHOLD = 0.35    # default UI slider value
NMS_IOU_THRESHOLD      = 0.4     # overlap threshold for duplicate removal

# ---------------------------------------------------------------------------
# SAR auto-detection
# ---------------------------------------------------------------------------
SAR_CHANNEL_DIFF_THRESHOLD = 8.0  # below this → image treated as grayscale/SAR

# ---------------------------------------------------------------------------
# COCO validator settings
# ---------------------------------------------------------------------------
COCO_VALIDATOR_CONF    = 0.25    # confidence for COCO second-stage check
COCO_REJECT_CONF       = 0.40    # COCO must be this sure to reject a detection
COCO_CROP_PADDING      = 0.10    # fractional padding around each crop
COCO_CONF_BOOST        = 0.10    # confidence boost when COCO confirms a boat

# COCO class IDs that are land/air objects — presence rejects the detection
COCO_NON_SHIP_CLASSES  = {
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    6,   # train
    7,   # truck
    14,  # bird
    15,  # cat
    16,  # dog
}
COCO_BOAT_CLASS        = 8       # confirms detection when present

# ---------------------------------------------------------------------------
# Ship keyword filter
# ---------------------------------------------------------------------------
SHIP_KEYWORDS = frozenset({
    "ship", "boat", "vessel", "tanker", "cargo",
    "carrier", "fishing", "tug", "ferry", "bulk",
})

# ---------------------------------------------------------------------------
# Vessel size/shape classification thresholds (pixels²)
# ---------------------------------------------------------------------------
VESSEL_LARGE_AREA      = 40000   # → Cargo / Tanker
VESSEL_MEDIUM_AREA     = 15000   # → Container Ship
VESSEL_SMALL_AREA      = 3000    # below this → Fishing Vessel
VESSEL_ELONGATED_RATIO = 3.5     # width/height ratio → Bulk Carrier

# ---------------------------------------------------------------------------
# Annotation colours (BGR) — one per detection index
# ---------------------------------------------------------------------------
BOX_COLORS = [
    (0, 255, 0),    # green
    (0, 165, 255),  # orange
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
]
