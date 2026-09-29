"""Phase 4.2 Crowd Intelligence & Image-Space Visual Density Analysis Service.

DISCLAIMER:
All density metrics produced by this service represent RELATIVE IMAGE-SPACE
OCCUPANCY and VIEWPOINT HEADCOUNT INDEXES, calculated from 2D camera detections
and spatial frame texture complexity.
They are NOT calibrated physical measurements of persons/m² or ground-plane density.
All thresholds and reference capacity values are PROVISIONAL PROTOTYPE VALUES
for system demonstration and testing, NOT validated crowd-safety engineering standards.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any
import cv2
import numpy as np

from config import (
    ANOMALY_DOWNGRADE_THRESHOLD,
    ANOMALY_ESCALATION_THRESHOLD,
    ANOMALY_LEVEL_CRITICAL,
    ANOMALY_LEVEL_ELEVATED,
    ANOMALY_LEVEL_NORMAL,
    ANOMALY_LEVEL_WATCH,
    ANOMALY_PERSISTENCE_FRAMES,
    ANOMALY_ROLLING_WINDOW_FRAMES,
    ANOMALY_WEIGHT_DECEL,
    ANOMALY_WEIGHT_REVERSAL,
    ANOMALY_WEIGHT_STAGNATION,
    ANOMALY_WEIGHT_SURGE,
    ANOMALY_WEIGHT_TURBULENCE,
    CONGESTION_HYSTERESIS_DOWNGRADE_THRESHOLD,
    CONGESTION_LEVEL_CRITICAL,
    CONGESTION_LEVEL_HIGH,
    CONGESTION_LEVEL_LOW,
    CONGESTION_LEVEL_MEDIUM,
    CONGESTION_PERSISTENCE_FRAMES,
    DENSITY_LEVEL_CRITICAL,
    DENSITY_LEVEL_HIGH,
    DENSITY_LEVEL_LOW,
    DENSITY_LEVEL_MEDIUM,
    DENSITY_SMOOTHING_ALPHA,
    FLOW_EMA_ALPHA,
    FLOW_MAX_DISPLACEMENT_PX,
    FLOW_MAX_TRACK_TIME_GAP,
    FLOW_NOMINAL_SPEED,
    FLOW_SPEED_UNIT,
    FLOW_STATIONARY_SPEED_THRESHOLD,
    FLOW_VELOCITY_DECAY,
    HYBRID_DENSITY_VISUAL_WEIGHT,
    HYBRID_DENSITY_YOLO_WEIGHT,
    HYBRID_RESIDUAL_BOOST_FACTOR,
    PROVISIONAL_ANOMALY_THRESHOLDS,
    PROVISIONAL_CONGESTION_THRESHOLDS,
    PROVISIONAL_DENSITY_THRESHOLDS,
    PROVISIONAL_OCCUPANCY_THRESHOLDS,
    PROVISIONAL_REFERENCE_CAPACITY,
    VISUAL_ANALYSIS_HEIGHT,
    VISUAL_ANALYSIS_WIDTH,
    VISUAL_CELL_LAPLACIAN_THRESHOLD,
    VISUAL_CELL_STD_THRESHOLD,
    VISUAL_GRID_COLS,
    VISUAL_GRID_ROWS,
    VISUAL_MAX_EXPECTED_OCCUPANCY,
)


@dataclass
class CrowdDensityResult:
    """Per-frame image-space crowd density metrics."""

    occupancy_ratio: float          # Fraction of frame area occupied by bounding box union (0.0 - 1.0)
    headcount_index: float          # Ratio of detected people to provisional reference capacity (0.0 - 1.0)
    density_index: float            # Composite image-space relative density index (0.0 - 1.0)
    smoothed_density_index: float   # Temporally smoothed density index via EMA (0.0 - 1.0)
    crowd_status: str               # LOW / MEDIUM / HIGH / CRITICAL
    reference_capacity: int         # Nominal capacity parameter for this camera viewpoint
    yolo_density: float = 0.0       # D_yolo component based purely on YOLO bounding boxes and headcount
    visual_density_index: float = 0.0  # D_visual component based on spatial grid texture clutter
    visual_occupancy: float = 0.0   # Ratio of spatial grid cells displaying crowd visual texture

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CrowdFlowCongestionResult:
    """Per-frame image-space crowd flow and congestion metrics."""

    flow_speed: float                   # Smoothed crowd speed in image-space pixels per second (px/s)
    flow_direction: str                 # Compass direction string (e.g., '→ East', '↘ South-East', 'STATIONARY')
    active_flow_vectors: int            # Number of active tracked movement vectors in this frame
    congestion_index: float             # Composite congestion metric in [0.0, 1.0] (Density * Stagnation)
    congestion_status: str              # Operational congestion status: LOW / MEDIUM / HIGH / CRITICAL
    flow_speed_unit: str = FLOW_SPEED_UNIT  # Explicit 'px/s' unit (not meters/second)
    smoothed_vx: float = 0.0            # Internal smoothed velocity component x (px/s)
    smoothed_vy: float = 0.0            # Internal smoothed velocity component y (px/s)
    stagnation_factor: float = 0.0      # Ratio of speed deficit (1.0 = stopped, 0.0 = brisk)
    high_congestion_counter: int = 0    # Persistent frame counter for HIGH/CRITICAL congestion
    individual_vectors: list[tuple[float, float]] = field(default_factory=list)  # Active vector pairs (vx, vy)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CrowdAnomalyResult:
    """Per-frame image-space crowd behaviour anomaly intelligence metrics."""

    anomaly_score: float             # Composite anomaly score in [0.0, 1.0]
    anomaly_status: str              # Operational behaviour status: NORMAL / WATCH / ELEVATED / CRITICAL
    anomaly_reason: str              # Primary human-readable operational explanation
    stagnation_anomaly: float = 0.0      # Component A_stagnation in [0.0, 1.0]
    density_surge_anomaly: float = 0.0   # Component A_surge in [0.0, 1.0]
    deceleration_anomaly: float = 0.0    # Component A_decel in [0.0, 1.0]
    reversal_anomaly: float = 0.0        # Component A_reversal in [0.0, 1.0]
    turbulence_anomaly: float = 0.0      # Component A_turbulence in [0.0, 1.0]
    high_anomaly_counter: int = 0        # Persistent consecutive frame counter for A >= 0.60

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_occupancy_ratio(detections: list, frame_width: int, frame_height: int) -> float:
    """Calculate the bounding box union area divided by total frame area.

    Uses a fast downscaled binary grid representation (16x spatial reduction)
    to compute exact multi-bounding-box overlap union in <0.1ms per frame.
    """
    if not detections or frame_width <= 0 or frame_height <= 0:
        return 0.0

    scale = 16
    grid_w = max(1, frame_width // scale)
    grid_h = max(1, frame_height // scale)

    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

    for det in detections:
        bbox = det.bbox if hasattr(det, "bbox") else det.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1 = max(0, min(grid_w, int(bbox[0] // scale)))
        y1 = max(0, min(grid_h, int(bbox[1] // scale)))
        x2 = max(0, min(grid_w, int(np.ceil(bbox[2] / scale))))
        y2 = max(0, min(grid_h, int(np.ceil(bbox[3] / scale))))

        if x2 > x1 and y2 > y1:
            grid[y1:y2, x1:x2] = 1

    return float(np.mean(grid))


def calculate_visual_density(frame: np.ndarray | None) -> tuple[float, float]:
    """Calculate image-space visual crowd occupancy and density index.

    Uses a spatial 16x9 grid on a 480x270 downscaled grayscale frame.
    Each cell is flagged as visually occupied if:
      mean(abs(Laplacian)) >= VISUAL_CELL_LAPLACIAN_THRESHOLD (7.5)
      AND
      grayscale StdDev >= VISUAL_CELL_STD_THRESHOLD (20.0)

    Returns:
      tuple: (visual_occupancy [0.0 - 1.0], visual_density_index [0.0 - 1.0])
    """
    if frame is None or not hasattr(frame, "shape") or frame.size == 0:
        return 0.0, 0.0

    try:
        small = cv2.resize(
            frame,
            (VISUAL_ANALYSIS_WIDTH, VISUAL_ANALYSIS_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        if len(small.shape) == 3:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            gray = small

        lap_abs = np.abs(cv2.Laplacian(gray, cv2.CV_32F))

        cell_h = VISUAL_ANALYSIS_HEIGHT // VISUAL_GRID_ROWS
        cell_w = VISUAL_ANALYSIS_WIDTH // VISUAL_GRID_COLS

        active_cells = 0
        total_cells = VISUAL_GRID_ROWS * VISUAL_GRID_COLS

        for r in range(VISUAL_GRID_ROWS):
            y_start = r * cell_h
            y_end = (r + 1) * cell_h
            for c in range(VISUAL_GRID_COLS):
                x_start = c * cell_w
                x_end = (c + 1) * cell_w

                cell_lap = lap_abs[y_start:y_end, x_start:x_end]
                cell_gray = gray[y_start:y_end, x_start:x_end]

                mean_lap = float(np.mean(cell_lap))
                std_gray = float(np.std(cell_gray))

                if (
                    mean_lap >= VISUAL_CELL_LAPLACIAN_THRESHOLD
                    and std_gray >= VISUAL_CELL_STD_THRESHOLD
                ):
                    active_cells += 1

        visual_occupancy = round(float(active_cells / max(1, total_cells)), 4)
        # Phase 4.2 Calibrated: D_visual = ActiveCells / 144 clamped to [0, 1]
        visual_density_index = round(
            min(1.0, max(0.0, float(active_cells / max(1, total_cells)))),
            4,
        )
        return visual_occupancy, visual_density_index

    except Exception:
        return 0.0, 0.0


def classify_density_status(density_index: float) -> str:
    """Classify relative image-space density into operational alert levels.

    Provisional prototype scale:
    - LOW:      < 0.25
    - MEDIUM:   0.25 <= index < 0.55
    - HIGH:     0.55 <= index < 0.80
    - CRITICAL: >= 0.80
    """
    if density_index < PROVISIONAL_DENSITY_THRESHOLDS["LOW"]:
        return DENSITY_LEVEL_LOW
    elif density_index < PROVISIONAL_DENSITY_THRESHOLDS["MEDIUM"]:
        return DENSITY_LEVEL_MEDIUM
    elif density_index < PROVISIONAL_DENSITY_THRESHOLDS["HIGH"]:
        return DENSITY_LEVEL_HIGH
    else:
        return DENSITY_LEVEL_CRITICAL


def calculate_frame_density(
    detections: list,
    frame: Any = None,
    frame_width: int = 1920,
    frame_height: int = 1080,
    previous_smoothed: float | None = None,
    reference_capacity: int = PROVISIONAL_REFERENCE_CAPACITY,
) -> CrowdDensityResult:
    """Compute complete image-space relative density metrics for a video frame.

    Phase 4.2 Hybrid Approach:
    - Retains D_yolo based on real YOLO detections and bounding box union area.
    - Computes D_visual from spatial 16x9 grid texture clutter on downscaled frame.
    - Fuses both signals via Conservative Residual Boost: D_hybrid = D_yolo + (1 - D_yolo) * (0.35 * D_visual)
    - Applies EMA smoothing for frame-to-frame stability.
    - Categorizes into LOW / MEDIUM / HIGH / CRITICAL.

    Args:
        detections: List of Detection objects (from PersonDetector).
        frame: Optional BGR numpy array of the current video frame.
        frame_width: Frame pixel width.
        frame_height: Frame pixel height.
        previous_smoothed: Prior frame's smoothed density index for EMA continuity.
        reference_capacity: Provisional nominal capacity for this viewpoint.

    Returns:
        CrowdDensityResult containing occupancy, headcount index, composite index,
        EMA smoothed index, and categorical status.
    """
    # Defensive argument handling for backwards compatibility
    if isinstance(frame, (int, float)):
        # Caller used old signature: calculate_frame_density(detections, width, height, prev, cap)
        reference_capacity = previous_smoothed or reference_capacity
        previous_smoothed = frame_height if isinstance(frame_height, float) else None
        frame_height = frame_width
        frame_width = int(frame)
        frame = None

    headcount = len(detections)

    # 1. Phase 4.1 YOLO detection density component (D_yolo)
    if headcount > 0 and frame_width > 0 and frame_height > 0:
        occupancy_ratio = calculate_occupancy_ratio(detections, frame_width, frame_height)
        occupancy_ratio = round(min(1.0, max(0.0, occupancy_ratio)), 4)
        headcount_index = round(
            min(1.0, max(0.0, headcount / max(1, reference_capacity))),
            4,
        )
        occupancy_norm = min(1.0, occupancy_ratio / PROVISIONAL_OCCUPANCY_THRESHOLDS["HIGH"])
        d_yolo = round(
            min(1.0, max(0.0, 0.5 * occupancy_norm + 0.5 * headcount_index)),
            4,
        )
    else:
        occupancy_ratio = 0.0
        headcount_index = 0.0
        d_yolo = 0.0

    # 2. Phase 4.2 Visual clutter density component (D_visual)
    visual_occupancy, visual_density_index = calculate_visual_density(frame)

    # 3. Hybrid fusion (Phase 4.2 Option C: Conservative Residual Boost):
    # D_hybrid = D_yolo + (1 - D_yolo) * (0.35 * D_visual) clamped to [0, 1]
    if visual_density_index > 0:
        d_hybrid = d_yolo + (1.0 - d_yolo) * (HYBRID_RESIDUAL_BOOST_FACTOR * visual_density_index)
    else:
        d_hybrid = d_yolo

    raw_density_index = round(min(1.0, max(0.0, d_hybrid)), 4)

    # 4. Temporal smoothing via EMA
    if previous_smoothed is None:
        smoothed_density_index = raw_density_index
    else:
        smoothed_density_index = round(
            DENSITY_SMOOTHING_ALPHA * raw_density_index
            + (1.0 - DENSITY_SMOOTHING_ALPHA) * previous_smoothed,
            4,
        )

    # Clean zero-snapping for completely empty scenes
    if raw_density_index == 0.0 and smoothed_density_index < 0.01:
        smoothed_density_index = 0.0

    crowd_status = classify_density_status(smoothed_density_index)

    return CrowdDensityResult(
        occupancy_ratio=occupancy_ratio,
        headcount_index=headcount_index,
        density_index=raw_density_index,
        smoothed_density_index=smoothed_density_index,
        crowd_status=crowd_status,
        reference_capacity=reference_capacity,
        yolo_density=d_yolo,
        visual_density_index=visual_density_index,
        visual_occupancy=visual_occupancy,
    )


def classify_flow_direction(vx: float, vy: float) -> str:
    """Map velocity vector in image coordinates to 8-point compass directions.

    Screen convention:
      +x = East / Right
      +y = South / Down
    """
    angle_deg = math.degrees(math.atan2(vy, vx))

    if -22.5 <= angle_deg < 22.5:
        return "→ East"
    elif 22.5 <= angle_deg < 67.5:
        return "↘ South-East"
    elif 67.5 <= angle_deg < 112.5:
        return "↓ South"
    elif 112.5 <= angle_deg < 157.5:
        return "↙ South-West"
    elif angle_deg >= 157.5 or angle_deg < -157.5:
        return "← West"
    elif -157.5 <= angle_deg < -112.5:
        return "↖ North-West"
    elif -112.5 <= angle_deg < -67.5:
        return "↑ North"
    else:
        return "↗ North-East"


def calculate_crowd_flow_and_congestion(
    detections: list,
    current_frame: int,
    current_timestamp: float,
    track_history: dict[int, tuple[float, float, int, float]],
    smoothed_density: float,
    previous_vx: float = 0.0,
    previous_vy: float = 0.0,
    previous_congestion_status: str = CONGESTION_LEVEL_LOW,
    high_congestion_counter: int = 0,
) -> tuple[CrowdFlowCongestionResult, dict[int, tuple[float, float, int, float]]]:
    """Calculate image-space crowd flow velocity, direction, and persistent congestion.

    Args:
        detections: Current frame detections with bounding boxes and optional track_ids.
        current_frame: Current frame number.
        current_timestamp: Current frame timestamp in seconds.
        track_history: In-memory cache mapping track_id -> (cx, cy, frame, timestamp).
        smoothed_density: Current EMA-smoothed image-space crowd density index.
        previous_vx: Prior frame smoothed velocity x (px/s).
        previous_vy: Prior frame smoothed velocity y (px/s).
        previous_congestion_status: Prior frame congestion status for hysteresis.
        high_congestion_counter: Number of consecutive frames sustaining HIGH/CRITICAL congestion.

    Returns:
        tuple of (CrowdFlowCongestionResult, updated_track_history).
    """
    new_history = dict(track_history)
    valid_vx_list = []
    valid_vy_list = []

    for det in detections:
        track_id = getattr(det, "track_id", None) if not isinstance(det, dict) else det.get("track_id")
        if track_id is None:
            continue

        bbox = getattr(det, "bbox", None) if not isinstance(det, dict) else det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        if track_id in track_history:
            prev_cx, prev_cy, prev_f, prev_t = track_history[track_id]
            dt = current_timestamp - prev_t
            dx = cx - prev_cx
            dy = cy - prev_cy
            displacement = math.hypot(dx, dy)

            # Reject invalid samples: dt outside (0, 1.5s] or displacement > 150 px
            if 0.0 < dt <= FLOW_MAX_TRACK_TIME_GAP and displacement <= FLOW_MAX_DISPLACEMENT_PX:
                valid_vx_list.append(dx / dt)
                valid_vy_list.append(dy / dt)

        new_history[track_id] = (cx, cy, current_frame, current_timestamp)

    # Clean up stale tracks from history (older than 10 seconds)
    stale_keys = [
        tid for tid, pos in new_history.items()
        if (current_timestamp - pos[3]) > 10.0
    ]
    for tid in stale_keys:
        new_history.pop(tid, None)

    active_flow_vectors = len(valid_vx_list)

    # Temporal velocity smoothing
    if active_flow_vectors > 0:
        mean_vx = float(np.mean(valid_vx_list))
        mean_vy = float(np.mean(valid_vy_list))
        smoothed_vx = FLOW_EMA_ALPHA * mean_vx + (1.0 - FLOW_EMA_ALPHA) * previous_vx
        smoothed_vy = FLOW_EMA_ALPHA * mean_vy + (1.0 - FLOW_EMA_ALPHA) * previous_vy
    else:
        # Graceful multiplicative decay when no track pairs exist in this frame
        smoothed_vx = previous_vx * FLOW_VELOCITY_DECAY
        smoothed_vy = previous_vy * FLOW_VELOCITY_DECAY

    flow_speed = math.hypot(smoothed_vx, smoothed_vy)
    if flow_speed < 0.001:
        flow_speed = 0.0
        smoothed_vx = 0.0
        smoothed_vy = 0.0

    # Flow direction classification
    if flow_speed < FLOW_STATIONARY_SPEED_THRESHOLD:
        flow_direction = "STATIONARY"
    else:
        flow_direction = classify_flow_direction(smoothed_vx, smoothed_vy)

    # Congestion calculation
    # Stagnation factor = 1.0 - (flow_speed / nominal_flow_speed) clamped to [0, 1]
    stagnation_factor = min(1.0, max(0.0, 1.0 - (flow_speed / max(1.0, FLOW_NOMINAL_SPEED))))

    # Congestion index = smoothed_density * stagnation_factor clamped to [0, 1]
    congestion_index = round(min(1.0, max(0.0, smoothed_density * stagnation_factor)), 4)

    # Congestion alert level classification with persistence and hysteresis
    if congestion_index < PROVISIONAL_CONGESTION_THRESHOLDS["LOW"]:
        raw_status = CONGESTION_LEVEL_LOW
    elif congestion_index < PROVISIONAL_CONGESTION_THRESHOLDS["MEDIUM"]:
        raw_status = CONGESTION_LEVEL_MEDIUM
    elif congestion_index < PROVISIONAL_CONGESTION_THRESHOLDS["HIGH"]:
        raw_status = CONGESTION_LEVEL_HIGH
    else:
        raw_status = CONGESTION_LEVEL_CRITICAL

    # Update persistence counter for high/critical levels
    if raw_status in (CONGESTION_LEVEL_HIGH, CONGESTION_LEVEL_CRITICAL):
        high_congestion_counter += 1
    else:
        high_congestion_counter = max(0, high_congestion_counter - 1)

    # Hysteresis and persistence resolution
    if previous_congestion_status in (CONGESTION_LEVEL_HIGH, CONGESTION_LEVEL_CRITICAL):
        # Downgrade from HIGH/CRITICAL only when congestion_index drops below hysteresis threshold (0.42)
        if congestion_index < CONGESTION_HYSTERESIS_DOWNGRADE_THRESHOLD:
            current_status = raw_status
        else:
            current_status = previous_congestion_status
    else:
        # Require sustained persistence frames (8 frames ~ 1.5s) before escalating to HIGH or CRITICAL
        if raw_status in (CONGESTION_LEVEL_HIGH, CONGESTION_LEVEL_CRITICAL):
            if high_congestion_counter >= CONGESTION_PERSISTENCE_FRAMES:
                current_status = raw_status
            else:
                current_status = CONGESTION_LEVEL_MEDIUM
        else:
            current_status = raw_status

    result = CrowdFlowCongestionResult(
        flow_speed=round(flow_speed, 2),
        flow_direction=flow_direction,
        active_flow_vectors=active_flow_vectors,
        congestion_index=congestion_index,
        congestion_status=current_status,
        flow_speed_unit=FLOW_SPEED_UNIT,
        smoothed_vx=round(smoothed_vx, 4),
        smoothed_vy=round(smoothed_vy, 4),
        stagnation_factor=round(stagnation_factor, 4),
        high_congestion_counter=high_congestion_counter,
        individual_vectors=list(zip(valid_vx_list, valid_vy_list)),
    )

    return result, new_history


def calculate_crowd_anomalies(
    smoothed_density: float,
    flow_speed: float,
    smoothed_vx: float,
    smoothed_vy: float,
    stagnation_factor: float,
    individual_vectors: list[tuple[float, float]],
    history_buffer: Any,
    previous_anomaly_status: str = ANOMALY_LEVEL_NORMAL,
    high_anomaly_counter: int = 0,
) -> CrowdAnomalyResult:
    """Calculate Phase 6 image-space crowd behaviour anomaly metrics.

    Evaluates 5 behavioral anomaly components:
    1. STAGNATION: smoothed_density * stagnation_factor when density >= 0.45 and speed < 5 px/s
    2. DENSITY SURGE: (smoothed_density - window_density_baseline) / 0.20 when density >= 0.40
    3. FLOW DECELERATION: (window_speed_baseline - flow_speed) / FLOW_NOMINAL_SPEED when baseline speed >= 30 and density >= 0.35
    4. COUNTER-FLOW REVERSAL: angular difference >= 110 deg when speed >= 15 and baseline speed >= 15
    5. DIRECTIONAL TURBULENCE: R_coherence < 0.40 when active_flow_vectors >= 3, speed >= 20, density >= 0.40

    Uses rolling window of up to 50 processed frames for baselines, 10-frame persistence,
    and 0.45/0.60 hysteresis.

    Args:
        smoothed_density: Current frame EMA-smoothed density index.
        flow_speed: Current frame smoothed flow speed (px/s).
        smoothed_vx: Current frame smoothed flow velocity x (px/s).
        smoothed_vy: Current frame smoothed flow velocity y (px/s).
        stagnation_factor: Stagnation ratio in [0.0, 1.0] from Phase 5.
        individual_vectors: List of (vx, vy) for active tracks in current frame.
        history_buffer: Iterable of past frame dicts containing:
            {'smoothed_density', 'flow_speed', 'smoothed_vx', 'smoothed_vy'}.
        previous_anomaly_status: Anomaly status from preceding frame for hysteresis.
        high_anomaly_counter: Number of consecutive frames sustaining score >= 0.60.

    Returns:
        CrowdAnomalyResult containing score, status, primary reason, and component values.
    """
    hist_list = list(history_buffer) if history_buffer else []

    if hist_list:
        window_density_baseline = float(np.mean([h.get("smoothed_density", smoothed_density) for h in hist_list]))
        window_speed_baseline = float(np.mean([h.get("flow_speed", flow_speed) for h in hist_list]))
        base_vx = float(np.mean([h.get("smoothed_vx", smoothed_vx) for h in hist_list]))
        base_vy = float(np.mean([h.get("smoothed_vy", smoothed_vy) for h in hist_list]))
    else:
        window_density_baseline = smoothed_density
        window_speed_baseline = flow_speed
        base_vx = smoothed_vx
        base_vy = smoothed_vy

    # 1. Stagnation Anomaly Component (A_stagnation)
    if smoothed_density >= 0.45 and flow_speed < FLOW_STATIONARY_SPEED_THRESHOLD:
        a_stagnation = min(1.0, max(0.0, smoothed_density * stagnation_factor))
    else:
        a_stagnation = 0.0

    # 2. Density Surge Component (A_surge)
    delta_density = max(0.0, smoothed_density - window_density_baseline)
    if smoothed_density >= 0.40:
        a_surge = min(1.0, max(0.0, delta_density / 0.20))
    else:
        a_surge = 0.0

    # 3. Flow Deceleration Component (A_decel)
    speed_drop = max(0.0, window_speed_baseline - flow_speed)
    if window_speed_baseline >= 30.0 and smoothed_density >= 0.35:
        a_decel = min(1.0, max(0.0, speed_drop / max(1.0, FLOW_NOMINAL_SPEED)))
    else:
        a_decel = 0.0

    # 4. Counter-Flow Reversal Component (A_reversal)
    cur_mag = math.hypot(smoothed_vx, smoothed_vy)
    base_mag = math.hypot(base_vx, base_vy)

    if flow_speed >= 15.0 and window_speed_baseline >= 15.0 and cur_mag > 0.001 and base_mag > 0.001:
        dot_prod = smoothed_vx * base_vx + smoothed_vy * base_vy
        cos_angle = min(1.0, max(-1.0, dot_prod / (cur_mag * base_mag)))
        angular_diff = math.degrees(math.acos(cos_angle))
        if angular_diff >= 110.0:
            a_reversal = min(1.0, max(0.0, (angular_diff - 90.0) / 90.0))
        else:
            a_reversal = 0.0
    else:
        a_reversal = 0.0

    # 5. Directional Turbulence Component (A_turbulence)
    m_vectors = len(individual_vectors) if individual_vectors else 0
    if m_vectors >= 3:
        sum_ux = 0.0
        sum_uy = 0.0
        indiv_speeds = []
        for vx, vy in individual_vectors:
            spd = math.hypot(vx, vy)
            indiv_speeds.append(spd)
            if spd > 0.001:
                sum_ux += vx / spd
                sum_uy += vy / spd
        r_coherence = math.hypot(sum_ux, sum_uy) / m_vectors
        mean_speed = float(np.mean(indiv_speeds)) if indiv_speeds else 0.0

        if r_coherence < 0.40 and mean_speed >= 20.0 and smoothed_density >= 0.40:
            a_turbulence = min(1.0, max(0.0, (0.40 - r_coherence) / 0.40))
        else:
            a_turbulence = 0.0
    else:
        a_turbulence = 0.0

    # Composite Anomaly Score
    a_raw = (
        ANOMALY_WEIGHT_STAGNATION * a_stagnation
        + ANOMALY_WEIGHT_SURGE * a_surge
        + ANOMALY_WEIGHT_DECEL * a_decel
        + ANOMALY_WEIGHT_REVERSAL * a_reversal
        + ANOMALY_WEIGHT_TURBULENCE * a_turbulence
    )
    anomaly_score = round(min(1.0, max(0.0, a_raw)), 4)

    # Behaviour status level determination
    if anomaly_score < PROVISIONAL_ANOMALY_THRESHOLDS["NORMAL"]:
        raw_status = ANOMALY_LEVEL_NORMAL
    elif anomaly_score < PROVISIONAL_ANOMALY_THRESHOLDS["WATCH"]:
        raw_status = ANOMALY_LEVEL_WATCH
    elif anomaly_score < PROVISIONAL_ANOMALY_THRESHOLDS["ELEVATED"]:
        raw_status = ANOMALY_LEVEL_ELEVATED
    else:
        raw_status = ANOMALY_LEVEL_CRITICAL

    # Update persistence counter for escalation conditions (score >= 0.60)
    if anomaly_score >= ANOMALY_ESCALATION_THRESHOLD:
        high_anomaly_counter += 1
    else:
        high_anomaly_counter = max(0, high_anomaly_counter - 1)

    # Hysteresis and persistence resolution
    if previous_anomaly_status in (ANOMALY_LEVEL_ELEVATED, ANOMALY_LEVEL_CRITICAL):
        # Downgrade only when anomaly_score drops below DE-ESCALATION THRESHOLD (0.45)
        if anomaly_score < ANOMALY_DOWNGRADE_THRESHOLD:
            current_status = raw_status
        else:
            current_status = previous_anomaly_status
    else:
        # Require 10 consecutive processed frames before escalating to ELEVATED or CRITICAL
        if raw_status in (ANOMALY_LEVEL_ELEVATED, ANOMALY_LEVEL_CRITICAL):
            if high_anomaly_counter >= ANOMALY_PERSISTENCE_FRAMES:
                current_status = raw_status
            else:
                current_status = ANOMALY_LEVEL_WATCH
        else:
            current_status = raw_status

    # Trigger reason priority logic
    if a_stagnation >= 0.50:
        anomaly_reason = "Prolonged flow stagnation under elevated density"
    elif a_surge >= 0.60:
        anomaly_reason = "Rapid crowd density surge"
    elif a_decel >= 0.60:
        anomaly_reason = "Sudden flow deceleration / blockage"
    elif a_reversal >= 0.60:
        anomaly_reason = "Counter-flow directional reversal detected"
    elif a_turbulence >= 0.60:
        anomaly_reason = "Conflicting crowd movement directions"
    else:
        anomaly_reason = "Normal movement patterns"

    return CrowdAnomalyResult(
        anomaly_score=anomaly_score,
        anomaly_status=current_status,
        anomaly_reason=anomaly_reason,
        stagnation_anomaly=round(a_stagnation, 4),
        density_surge_anomaly=round(a_surge, 4),
        deceleration_anomaly=round(a_decel, 4),
        reversal_anomaly=round(a_reversal, 4),
        turbulence_anomaly=round(a_turbulence, 4),
        high_anomaly_counter=high_anomaly_counter,
    )

