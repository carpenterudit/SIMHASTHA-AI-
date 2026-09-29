from dataclasses import dataclass
from pathlib import Path

API_PREFIX = "/api"
SYSTEM_NAME = "Simhastha AI"
BACKEND_ROOT = Path(__file__).resolve().parent
DATA_ROOT = BACKEND_ROOT / "data"
UPLOADS_DIR = DATA_ROOT / "uploads"
PROCESSED_DIR = DATA_ROOT / "processed"
RESULTS_DIR = DATA_ROOT / "results"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_CONFIDENCE = 0.40
DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_INPUT_SIZE = 640
DEFAULT_FRAME_SKIP = 1

# Phase 4.1: Image-space relative crowd density constants
# NOTE: These values are provisional prototype parameters for demonstration, NOT validated crowd-safety standards.
# Density metrics represent relative image-space occupancy and viewpoint headcount, NOT physical persons/m².
DENSITY_LEVEL_LOW = "LOW"
DENSITY_LEVEL_MEDIUM = "MEDIUM"
DENSITY_LEVEL_HIGH = "HIGH"
DENSITY_LEVEL_CRITICAL = "CRITICAL"

# Provisional image-space bounding box occupancy thresholds (fraction of frame covered)
PROVISIONAL_OCCUPANCY_THRESHOLDS = {
    "LOW": 0.08,        # < 8% frame area coverage
    "MEDIUM": 0.20,     # 8% to 20% frame area coverage
    "HIGH": 0.35,       # 20% to 35% frame area coverage
                        # > 35% = CRITICAL
}

# Provisional reference nominal capacity for the primary camera viewpoint (Camera-07)
PROVISIONAL_REFERENCE_CAPACITY = 40

# Provisional composite density index thresholds (0.0 to 1.0 relative scale)
PROVISIONAL_DENSITY_THRESHOLDS = {
    "LOW": 0.25,        # < 0.25
    "MEDIUM": 0.55,     # 0.25 to 0.55
    "HIGH": 0.80,       # 0.55 to 0.80
                        # >= 0.80 = CRITICAL
}

# Exponential Moving Average (EMA) smoothing factor for temporal stability across frames
DENSITY_SMOOTHING_ALPHA = 0.25

# Phase 4.2: Visual crowd density analysis parameters (downscale resolution & spatial grid)
VISUAL_ANALYSIS_WIDTH = 480
VISUAL_ANALYSIS_HEIGHT = 270
VISUAL_GRID_ROWS = 9
VISUAL_GRID_COLS = 16
VISUAL_CELL_LAPLACIAN_THRESHOLD = 7.5
VISUAL_CELL_STD_THRESHOLD = 20.0
VISUAL_TOTAL_GRID_CELLS = 144  # 16 x 9 spatial cells
VISUAL_MAX_EXPECTED_OCCUPANCY = 1.0  # Normalized across full 144 cells (ActiveCells / 144)

# Phase 4.2 Option C: Conservative Residual Boost parameter:
# D_hybrid = D_yolo + (1 - D_yolo) * (HYBRID_RESIDUAL_BOOST_FACTOR * D_visual)
HYBRID_RESIDUAL_BOOST_FACTOR = 0.35

# Legacy weights preserved for backwards compatibility
HYBRID_DENSITY_YOLO_WEIGHT = 0.4
HYBRID_DENSITY_VISUAL_WEIGHT = 0.6

# Reference zone monitored by default CCTV camera (Camera-07)
DEFAULT_CAMERA_ZONE = "Ramghat"

# Phase 5: Crowd Flow and Congestion Analysis Parameters
# NOTE: Speeds are measured in relative 2D image-space pixels per second (px/s), NOT calibrated meters/second.
FLOW_NOMINAL_SPEED = 60.0              # Nominal unimpeded walking speed in image space (px/s)
FLOW_EMA_ALPHA = 0.20                  # Temporal EMA smoothing factor for velocity vector
FLOW_VELOCITY_DECAY = 0.92             # Multiplicative decay when no track pairs exist in a frame
FLOW_MAX_TRACK_TIME_GAP = 1.5          # Maximum allowed time delta between track observations (seconds)
FLOW_MAX_DISPLACEMENT_PX = 150.0       # Maximum plausible inter-frame displacement to reject anomalies (pixels)
FLOW_STATIONARY_SPEED_THRESHOLD = 5.0  # Speed below which flow is classified as STATIONARY (px/s)
FLOW_SPEED_UNIT = "px/s"               # Explicit image-space unit identifier

# Phase 5: Congestion Level Definitions & Thresholds
CONGESTION_LEVEL_LOW = "LOW"
CONGESTION_LEVEL_MEDIUM = "MEDIUM"
CONGESTION_LEVEL_HIGH = "HIGH"
CONGESTION_LEVEL_CRITICAL = "CRITICAL"

PROVISIONAL_CONGESTION_THRESHOLDS = {
    "LOW": 0.25,                       # C < 0.25: Free-flowing / low congestion
    "MEDIUM": 0.50,                    # 0.25 <= C < 0.50: Moderate crowd density / slow movement
    "HIGH": 0.75,                      # 0.50 <= C < 0.75: Significant congestion / bottleneck
                                       # >= 0.75: CRITICAL congestion / near-stationary crush hazard
}

# Persistence & Hysteresis to prevent noisy single-frame alarms
CONGESTION_PERSISTENCE_FRAMES = 8       # Approx 1.5 seconds sustained condition for HIGH/CRITICAL
CONGESTION_HYSTERESIS_DOWNGRADE_THRESHOLD = 0.42  # De-escalate from HIGH only when index drops below 0.42

# Phase 6: Image-Space Crowd Behaviour Anomaly Intelligence Parameters
# NOTE: These metrics represent relative image-space behavioural deviations, NOT certified real-world emergency predictions.
ANOMALY_ROLLING_WINDOW_FRAMES = 50          # Rolling temporal history window (50 processed frames ~ 2.0s)
ANOMALY_PERSISTENCE_FRAMES = 10             # Minimum consecutive frames above escalation condition (0.60) before alerting
ANOMALY_ESCALATION_THRESHOLD = 0.60         # Anomaly score required to trigger ELEVATED status
ANOMALY_DOWNGRADE_THRESHOLD = 0.45          # Hysteresis de-escalation threshold to downgrade from ELEVATED/CRITICAL

# Provisional component weights summing to 1.0
ANOMALY_WEIGHT_STAGNATION = 0.30            # Weight for stagnation under density
ANOMALY_WEIGHT_SURGE = 0.30                 # Weight for rapid density surge
ANOMALY_WEIGHT_DECEL = 0.20                 # Weight for sudden flow deceleration
ANOMALY_WEIGHT_REVERSAL = 0.10              # Weight for counter-flow directional reversal
ANOMALY_WEIGHT_TURBULENCE = 0.10            # Weight for multi-track directional turbulence

# Behaviour status level constants
ANOMALY_LEVEL_NORMAL = "NORMAL"
ANOMALY_LEVEL_WATCH = "WATCH"
ANOMALY_LEVEL_ELEVATED = "ELEVATED"
ANOMALY_LEVEL_CRITICAL = "CRITICAL"

PROVISIONAL_ANOMALY_THRESHOLDS = {
    "NORMAL": 0.30,                         # 0.00 <= A < 0.30: Normal crowd behaviour
    "WATCH": 0.60,                          # 0.30 <= A < 0.60: Monitoring condition / minor slowdown
    "ELEVATED": 0.80,                       # 0.60 <= A < 0.80: Persistent stagnation or rapid surge
                                            # A >= 0.80: CRITICAL multi-anomaly convergence
}

# Phase 7: Spatial Zone Intelligence Configuration
# Normalized coordinates: [x_min, y_min, x_max, y_max] where each value is in [0.0, 1.0].
@dataclass(frozen=True)
class ZoneDefinition:
    zone_id: str
    name: str
    rect: tuple[float, float, float, float]
    reference_capacity: int = 10


DEFAULT_ZONE_CONFIG_KEY = "camera_07"

CAMERA_ZONE_CONFIGS: dict[str, list[ZoneDefinition]] = {
    "camera_07": [
        ZoneDefinition(
            zone_id="zone_a",
            name="Zone A",
            rect=(0.0, 0.0, 0.5, 0.5),
            reference_capacity=10,
        ),
        ZoneDefinition(
            zone_id="zone_b",
            name="Zone B",
            rect=(0.5, 0.0, 1.0, 0.5),
            reference_capacity=10,
        ),
        ZoneDefinition(
            zone_id="zone_c",
            name="Zone C",
            rect=(0.0, 0.5, 0.5, 1.0),
            reference_capacity=10,
        ),
        ZoneDefinition(
            zone_id="zone_d",
            name="Zone D",
            rect=(0.5, 0.5, 1.0, 1.0),
            reference_capacity=10,
        ),
    ],
}


def get_camera_zones(camera_id: str | None = None) -> list[ZoneDefinition]:
    """Retrieve the list of configured zones for a camera viewpoint.

    If camera_id is None, empty, or unknown, safely falls back to DEFAULT_ZONE_CONFIG_KEY ('camera_07').
    """
    if camera_id and camera_id in CAMERA_ZONE_CONFIGS:
        return CAMERA_ZONE_CONFIGS[camera_id]
    return CAMERA_ZONE_CONFIGS[DEFAULT_ZONE_CONFIG_KEY]


for directory in (UPLOADS_DIR, PROCESSED_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


