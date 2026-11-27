#!/usr/bin/env python3
"""Convert PTB-XL+ 12SL median-beat WFDB records to standardized PNG images.

Input
-----
An extracted PTB-XL+ v1.0.1 12SL folder containing paired files such as:
    00000/00001_medians.hea
    00000/00001_medians.dat

Processing
----------
1. Parse the WFDB header and verify the expected 12 leads, 500 Hz, 600 samples,
   and WFDB format 32.
2. Decode little-endian, signed, 32-bit multiplexed samples.
3. Validate binary size, initial values and per-lead WFDB checksums.
4. Treat only INT32_MIN (-2147483648) as the standard format-32 invalid marker.
5. Preserve baseline-valued samples. Do not interpolate invalid samples.
6. Convert header-scaled values to the common mV scale using divisor 1000.
   This divisor was empirically cross-checked against the paired PTB-XL+ 12SL
   amplitude feature columns (R_Amp_*, Q_Amp_*, S_Amp_*) over all 20,914
   available 12SL waveform records. It is not an independent laboratory
   calibration. The optional --features-csv argument reproduces a per-run
   comparison and writes a calibration validation summary.
7. Render a fixed 4-row x 3-column 12-lead PNG with a common amplitude range.
8. Write a manifest, per-lead quality report, failures list and JSON config.

Important interpretation
------------------------
- These are 1.2-second median-beat templates, not full 10-second ECG traces.
- The PNGs are standardized visualization images. The grid is not certified
  clinical ECG paper at a physically exact pixel/mm aspect ratio.
- Technical flags identify invalid markers or display clipping; they are not
  clinical ECG-quality labels and must not be interpreted as diagnoses.
- The converter does not attach diagnostic labels or patient-level data splits.

Dependencies
------------
    python -m pip install numpy pandas pillow tqdm

Examples (run from your project root)
-------------------------------------
Test ten records:
    python3 convert_12sl_to_images_final.py --max-records 10

Convert all records:
    python3 convert_12sl_to_images_final.py --all

Convert all records and reproduce the feature-based scale check:
    python3 convert_12sl_to_images_final.py --all \\
        --features-csv data/ptb-xl-plus/features/12sl_features.csv

If paths differ, supply --input-dir, --output-dir and --features-csv explicitly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Dataset and output defaults (paths are relative to the current project root)
# -----------------------------------------------------------------------------
DEFAULT_INPUT_DIR = Path("data/ptb-xl-plus/median_beats/12sl")
DEFAULT_OUTPUT_DIR = Path("data/processed/12sl_median_images_final")

# PTB-XL+ 12SL median-beat layout
EXPECTED_FS_HZ = 500.0
EXPECTED_SAMPLES = 600
EXPECTED_LEADS = 12
EXPECTED_DURATION_S = EXPECTED_SAMPLES / EXPECTED_FS_HZ

# WFDB format 32 is signed 32-bit little-endian data in this release.
RAW_DTYPE = np.dtype("<i4")
WFDB_INVALID_SAMPLE = int(np.iinfo(np.int32).min)
INT32_MIN_PLUS_ONE = WFDB_INVALID_SAMPLE + 1
INT32_MAX = int(np.iinfo(np.int32).max)

# Empirically validated conversion to the mV scale used by the 12SL feature CSV.
# See the module docstring and the optional --features-csv comparison report.
DISPLAY_DIVISOR = 1000.0

# Fixed image settings. A +/-9 mV range covers all but one extreme record in
# the supplied archive; any actual clipping is recorded in the manifest.
IMAGE_WIDTH = 1200
IMAGE_HEIGHT = 1200
TRACE_WIDTH = 2
DISPLAY_Y_LIMIT_MV = 9.0
DISPLAY_Y_GRID_STEP_MV = 0.2
TIME_GRID_STEP_SECONDS = 0.04

# Long constant-baseline runs are advisory only; they are not invalid markers.
LONG_BASELINE_RUN_REVIEW_SAMPLES = 100

LEAD_ORDER = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3",
    "V4", "V5", "V6",
]

# Calibration comparison uses these fields from the standardized 12SL feature CSV.
AMPLITUDE_FEATURE_TYPES = ("R_Amp", "Q_Amp", "S_Amp")


# -----------------------------------------------------------------------------
# Header parsing
# -----------------------------------------------------------------------------
def parse_gain_baseline_units(token: str, adc_zero_text: str) -> tuple[float, float, str]:
    """Parse a WFDB adc_gain(baseline)/units token."""
    match = re.fullmatch(
        r"([0-9.eE+-]+)(?:\(([-+0-9.eE]+)\))?/(.+)",
        token,
    )
    if not match:
        raise ValueError(f"Cannot parse WFDB gain/baseline/unit token: {token!r}")

    gain = float(match.group(1))
    baseline = (
        float(match.group(2))
        if match.group(2) is not None
        else float(adc_zero_text)
    )
    units = match.group(3).strip()

    if not np.isfinite(gain) or gain <= 0:
        raise ValueError(f"Invalid ADC gain: {gain}")
    if not np.isfinite(baseline):
        raise ValueError(f"Invalid ADC baseline: {baseline}")
    if not np.isclose(baseline, round(baseline), atol=1e-6):
        raise ValueError(f"Expected integral ADC baseline, found {baseline}")
    if not units:
        raise ValueError("Empty physical unit in WFDB header")

    return gain, baseline, units


def parse_header(header_path: Path) -> dict[str, Any]:
    """Parse a 12SL median-beat WFDB header without resolving its stale path."""
    lines = [
        line.strip()
        for line in header_path.read_text(encoding="ascii").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError("Empty WFDB header")

    record_fields = lines[0].split()
    if len(record_fields) < 4:
        raise ValueError(f"Malformed WFDB record line: {lines[0]!r}")

    n_sig = int(record_fields[1])
    fs = float(record_fields[2])
    n_samples = int(record_fields[3])

    if n_sig != EXPECTED_LEADS:
        raise ValueError(f"Expected {EXPECTED_LEADS} leads, found {n_sig}")
    if not np.isfinite(fs) or not np.isclose(fs, EXPECTED_FS_HZ):
        raise ValueError(f"Expected {EXPECTED_FS_HZ} Hz, found {fs} Hz")
    if n_samples != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} samples/lead, found {n_samples}")
    if len(lines) < n_sig + 1:
        raise ValueError(f"Expected {n_sig} signal lines, found {len(lines) - 1}")

    signals: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines[1 : 1 + n_sig], start=2):
        fields = line.split()
        if len(fields) < 9:
            raise ValueError(f"Malformed signal line {line_number}: {line!r}")

        try:
            fmt = int(fields[1])
        except ValueError as exc:
            raise ValueError(f"Unsupported WFDB format token: {fields[1]!r}") from exc
        if fmt != 32:
            raise ValueError(f"Expected WFDB format 32, found {fmt}")

        gain, baseline, units = parse_gain_baseline_units(fields[2], fields[4])
        adc_resolution = int(fields[3])
        if adc_resolution != 32:
            raise ValueError(f"Expected 32-bit ADC resolution, found {adc_resolution}")

        signals.append(
            {
                "data_filename_in_header": fields[0],
                "format": fmt,
                "gain": gain,
                "baseline": baseline,
                "units": units,
                "adc_resolution": adc_resolution,
                "adc_zero": int(fields[4]),
                "init_value": int(fields[5]),
                "checksum": int(fields[6]),
                "block_size": int(fields[7]),
                "lead_name": fields[8],
            }
        )

    lead_names = [row["lead_name"] for row in signals]
    if len(set(lead_names)) != EXPECTED_LEADS:
        raise ValueError(f"Duplicate lead names in header: {lead_names}")
    if set(lead_names) != set(LEAD_ORDER):
        raise ValueError(f"Unexpected lead-name set: {lead_names}")

    unit_set = sorted({row["units"] for row in signals})
    if len(unit_set) != 1:
        raise ValueError(f"Mixed header units across leads: {unit_set}")

    return {
        "record_name_in_header": record_fields[0],
        "n_sig": n_sig,
        "fs": fs,
        "n_samples": n_samples,
        "signals": signals,
        "header_units": unit_set[0],
    }


def get_ecg_id(header_path: Path) -> int:
    """Extract the ECG identifier from e.g. 00004_medians.hea."""
    match = re.fullmatch(r"(\d+)_medians", header_path.stem)
    if not match:
        raise ValueError(f"Unexpected 12SL header filename: {header_path.name}")
    return int(match.group(1))


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def longest_true_run(mask: np.ndarray) -> int:
    """Return the maximum consecutive run of True values in a 1-D mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("longest_true_run expects a one-dimensional mask")
    if not mask.any():
        return 0
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return int(np.max(ends - starts)) if starts.size else 0


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a common Linux font, falling back to Pillow's default font."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for filename in candidates:
        if Path(filename).is_file():
            try:
                return ImageFont.truetype(filename, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_waveform_segments(
    draw: ImageDraw.ImageDraw,
    x_values: np.ndarray,
    y_values: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    """Draw contiguous valid sections separately so missing samples are not bridged."""
    edges = np.diff(np.r_[False, valid_mask, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    for start, end in zip(starts, ends):
        points = [
            (int(round(x)), int(round(y)))
            for x, y in zip(x_values[start:end], y_values[start:end])
        ]
        if len(points) >= 2:
            draw.line(points, fill=(15, 15, 15), width=TRACE_WIDTH)
        elif len(points) == 1:
            draw.point(points[0], fill=(15, 15, 15))


# -----------------------------------------------------------------------------
# Optional comparison against the paired 12SL amplitude feature columns
# -----------------------------------------------------------------------------
def load_feature_reference(features_csv: Path | None) -> pd.DataFrame | None:
    """Load only ecg_id and the 36 per-lead R/Q/S amplitude feature columns."""
    if features_csv is None:
        return None
    if not features_csv.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {features_csv}")

    required = ["ecg_id"] + [
        f"{feature}_{lead}"
        for lead in LEAD_ORDER
        for feature in AMPLITUDE_FEATURE_TYPES
    ]
    frame = pd.read_csv(features_csv, usecols=lambda col: col in set(required))
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(
            "The supplied feature CSV is missing required amplitude columns: "
            + ", ".join(missing)
        )
    if frame["ecg_id"].duplicated().any():
        raise ValueError("Duplicate ecg_id values found in the feature CSV")

    frame["ecg_id"] = frame["ecg_id"].astype(int)
    return frame.set_index("ecg_id", drop=True)


def add_calibration_comparisons(
    accumulator: dict[tuple[str, str], dict[str, list[float]]],
    header_scaled_signal: np.ndarray,
    lead_names: list[str],
    feature_row: pd.Series | None,
) -> None:
    """Accumulate R-peak and negative Q/S-extreme comparisons for two scales.

    header_scaled_signal is (digital - baseline) / gain, before the divisor.
    Feature columns R_Amp_*, Q_Amp_* and S_Amp_* in 12sl_features.csv are in
    the common mV scale. This is a robust consistency check, not a clinical
    validation and not an assertion that global extrema always equal named peaks.
    """
    if feature_row is None:
        return

    for lead_index, lead in enumerate(lead_names):
        values = header_scaled_signal[:, lead_index]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        # Positive R peak: compare only if the feature indicates a meaningful R wave.
        r_ref = feature_row.get(f"R_Amp_{lead}", np.nan)
        if pd.notna(r_ref) and np.isfinite(float(r_ref)) and float(r_ref) > 0.03:
            ref = float(r_ref)
            waveform_header_peak = float(np.max(values))
            bucket = accumulator.setdefault((lead, "positive_max_vs_R_Amp"), {"div1": [], "div1000": []})
            bucket["div1"].append(abs(waveform_header_peak - ref))
            bucket["div1000"].append(abs(waveform_header_peak / DISPLAY_DIVISOR - ref))

        # The most negative Q or S feature should be comparable to the negative
        # waveform extreme when the corresponding feature is present.
        negative_refs = []
        for feature_name in (f"Q_Amp_{lead}", f"S_Amp_{lead}"):
            value = feature_row.get(feature_name, np.nan)
            if pd.notna(value) and np.isfinite(float(value)) and float(value) < -0.03:
                negative_refs.append(float(value))
        if negative_refs:
            ref = min(negative_refs)
            waveform_header_min = float(np.min(values))
            bucket = accumulator.setdefault((lead, "negative_min_vs_min_QS"), {"div1": [], "div1000": []})
            bucket["div1"].append(abs(waveform_header_min - ref))
            bucket["div1000"].append(abs(waveform_header_min / DISPLAY_DIVISOR - ref))


def build_calibration_summary(
    accumulator: dict[tuple[str, str], dict[str, list[float]]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize candidate scale errors without silently changing the divisor."""
    rows: list[dict[str, Any]] = []
    all_div1: list[float] = []
    all_div1000: list[float] = []

    for (lead, comparison), values in sorted(accumulator.items()):
        e1 = np.asarray(values["div1"], dtype=float)
        e1000 = np.asarray(values["div1000"], dtype=float)
        if e1.size == 0:
            continue
        all_div1.extend(e1.tolist())
        all_div1000.extend(e1000.tolist())
        rows.append(
            {
                "lead": lead,
                "comparison": comparison,
                "n_comparisons": int(e1.size),
                "median_abs_error_no_divisor": float(np.median(e1)),
                "p90_abs_error_no_divisor": float(np.quantile(e1, 0.90)),
                "median_abs_error_divide_1000": float(np.median(e1000)),
                "p90_abs_error_divide_1000": float(np.quantile(e1000, 0.90)),
                "fraction_within_0_02_mV_divide_1000": float(np.mean(e1000 <= 0.02)),
            }
        )

    if all_div1:
        a = np.asarray(all_div1, dtype=float)
        b = np.asarray(all_div1000, dtype=float)
        summary = {
            "n_feature_comparisons": int(b.size),
            "median_abs_error_no_divisor": float(np.median(a)),
            "median_abs_error_divide_1000": float(np.median(b)),
            "p90_abs_error_divide_1000": float(np.quantile(b, 0.90)),
            "fraction_within_0_02_mV_divide_1000": float(np.mean(b <= 0.02)),
            "empirical_result": (
                "supports_divisor_1000"
                if b.size >= 100 and np.median(b) < 0.03 and np.median(a) > 10 * max(np.median(b), 1e-9)
                else "inconclusive_or_needs_review"
            ),
            "interpretation": (
                "Compares waveform positive maxima with R_Amp and negative minima "
                "with the more negative Q_Amp/S_Amp feature. Some differences are "
                "expected because a global waveform extreme is not always the exact "
                "annotated wave peak. This is an empirical consistency check, not "
                "independent laboratory calibration."
            ),
        }
    else:
        summary = {
            "n_feature_comparisons": 0,
            "empirical_result": "not_run_no_features_csv_supplied",
            "interpretation": "Supply --features-csv to generate the per-run scale comparison.",
        }

    return pd.DataFrame(rows), summary


# -----------------------------------------------------------------------------
# Decode and validate one record
# -----------------------------------------------------------------------------
def decode_record(
    header_path: Path,
) -> tuple[int, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Decode one format-32 WFDB record.

    Returns:
      ecg_id, empirically scaled mV signal, header-scaled signal, details, lead rows
    """
    metadata = parse_header(header_path)
    data_path = header_path.with_suffix(".dat")
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing .dat file paired with {header_path}")

    expected_values = metadata["n_samples"] * metadata["n_sig"]
    expected_bytes = expected_values * RAW_DTYPE.itemsize
    actual_bytes = data_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"Expected {expected_bytes} bytes, found {actual_bytes}")

    raw_flat = np.fromfile(data_path, dtype=RAW_DTYPE)
    if raw_flat.size != expected_values:
        raise ValueError(f"Expected {expected_values} integers, found {raw_flat.size}")

    # The files are multiplexed: sample 0 of all 12 leads, then sample 1, etc.
    raw = raw_flat.reshape(metadata["n_samples"], metadata["n_sig"])

    # Verify first sample and per-lead 16-bit WFDB checksum.
    for lead_index, info in enumerate(metadata["signals"]):
        actual_first = int(raw[0, lead_index])
        if actual_first != info["init_value"]:
            raise ValueError(
                f"Initial-value mismatch in {info['lead_name']}: "
                f"header={info['init_value']}, data={actual_first}"
            )
        calculated_checksum = int(np.sum(raw[:, lead_index], dtype=np.int64)) & 0xFFFF
        expected_checksum = int(info["checksum"]) & 0xFFFF
        if calculated_checksum != expected_checksum:
            raise ValueError(
                f"Checksum mismatch in {info['lead_name']}: "
                f"header={expected_checksum}, calculated={calculated_checksum}"
            )

    # Exact format-32 invalid marker only. Do not use a broad near-limit test.
    signal_header_units = np.full(raw.shape, np.nan, dtype=np.float64)
    invalid_all = raw == WFDB_INVALID_SAMPLE
    lead_quality_rows: list[dict[str, Any]] = []

    for lead_index, info in enumerate(metadata["signals"]):
        digital = raw[:, lead_index]
        invalid = invalid_all[:, lead_index]
        valid = ~invalid
        baseline_digital = int(round(info["baseline"]))
        baseline_mask = (digital == baseline_digital) & valid

        digital_float = digital.astype(np.float64)
        signal_header_units[valid, lead_index] = (
            digital_float[valid] - info["baseline"]
        ) / info["gain"]

        # These codes are recorded for audit only. They are not automatically
        # converted to NaN or treated as invalid samples.
        near_rail = ((digital == INT32_MIN_PLUS_ONE) | (digital == INT32_MAX)) & valid
        near_rail_nonbaseline = near_rail & ~baseline_mask

        finite_header = signal_header_units[:, lead_index]
        finite_header = finite_header[np.isfinite(finite_header)]

        lead_quality_rows.append(
            {
                "lead": info["lead_name"],
                "header_units_token": info["units"],
                "adc_gain": info["gain"],
                "baseline_digital": baseline_digital,
                "invalid_samples": int(invalid.sum()),
                "invalid_fraction": float(invalid.mean()),
                "max_invalid_gap_samples": longest_true_run(invalid),
                "baseline_equal_samples": int(baseline_mask.sum()),
                "max_baseline_run_samples": longest_true_run(baseline_mask),
                "near_int32_rail_samples": int(near_rail.sum()),
                "near_rail_nonbaseline_samples": int(near_rail_nonbaseline.sum()),
                "header_value_min": float(finite_header.min()) if finite_header.size else float("nan"),
                "header_value_max": float(finite_header.max()) if finite_header.size else float("nan"),
            }
        )

    signal_mV = (signal_header_units / DISPLAY_DIVISOR).astype(np.float32)
    ecg_id = get_ecg_id(header_path)
    lead_names = [row["lead_name"] for row in metadata["signals"]]

    details = {
        "sampling_rate_hz": metadata["fs"],
        "samples_per_lead": metadata["n_samples"],
        "n_leads": metadata["n_sig"],
        "duration_seconds": metadata["n_samples"] / metadata["fs"],
        "lead_names": lead_names,
        "header_units": metadata["header_units"],
        "wfdb_invalid_samples": int(invalid_all.sum()),
        "wfdb_invalid_fraction": float(invalid_all.mean()),
        "max_invalid_gap_samples": max(
            (row["max_invalid_gap_samples"] for row in lead_quality_rows), default=0
        ),
        "max_baseline_run_samples": max(
            (row["max_baseline_run_samples"] for row in lead_quality_rows), default=0
        ),
        "n_leads_with_long_baseline_run": sum(
            row["max_baseline_run_samples"] > LONG_BASELINE_RUN_REVIEW_SAMPLES
            for row in lead_quality_rows
        ),
        "near_int32_rail_samples": sum(row["near_int32_rail_samples"] for row in lead_quality_rows),
        "near_rail_nonbaseline_samples": sum(
            row["near_rail_nonbaseline_samples"] for row in lead_quality_rows
        ),
        "max_abs_mV": float(np.nanmax(np.abs(signal_mV))) if np.isfinite(signal_mV).any() else float("nan"),
    }
    return ecg_id, signal_mV, signal_header_units, details, lead_quality_rows


# -----------------------------------------------------------------------------
# Render one 12-lead image
# -----------------------------------------------------------------------------
def render_12lead_image(
    signal_mV: np.ndarray,
    lead_names: list[str],
    ecg_id: int,
    sampling_rate_hz: float,
    output_path: Path,
) -> tuple[float, float]:
    """Render a fixed 4x3 image; return clipped fraction and maximum |mV|."""
    if signal_mV.ndim != 2 or signal_mV.shape[1] != EXPECTED_LEADS:
        raise ValueError(f"Expected (samples, 12) signal, got {signal_mV.shape}")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError(f"Invalid sampling rate: {sampling_rate_hz}")

    name_to_index = {name: idx for idx, name in enumerate(lead_names)}
    if len(name_to_index) != EXPECTED_LEADS or any(name not in name_to_index for name in LEAD_ORDER):
        raise ValueError(f"Missing or duplicate lead names: {lead_names}")
    signal_mV = signal_mV[:, [name_to_index[name] for name in LEAD_ORDER]]

    finite_mask = np.isfinite(signal_mV)
    finite_values = signal_mV[finite_mask]
    if finite_values.size == 0:
        raise ValueError(f"ECG {ecg_id} contains no valid samples")

    clipped_mask = finite_mask & (np.abs(signal_mV) > DISPLAY_Y_LIMIT_MV)
    clipped_fraction = float(clipped_mask.sum() / finite_mask.sum())
    max_abs_mV = float(np.max(np.abs(finite_values)))

    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(25)
    lead_font = load_font(19)
    small_font = load_font(14)

    draw.text(
        (24, 12),
        f"PTB-XL+ 12SL Median Beat | ECG ID: {ecg_id}",
        fill=(20, 20, 20),
        font=title_font,
    )
    draw.text(
        (25, 45),
        "12 leads | 500 Hz | 1.2 s | amplitude scale cross-checked with 12SL features",
        fill=(80, 80, 80),
        font=small_font,
    )

    margin_x, top, bottom = 20, 72, 18
    gap_x, gap_y = 12, 10
    panel_w = (IMAGE_WIDTH - 2 * margin_x - 2 * gap_x) // 3
    panel_h = (IMAGE_HEIGHT - top - bottom - 3 * gap_y) // 4

    duration_seconds = signal_mV.shape[0] / sampling_rate_hz
    n_x_small = max(1, int(round(duration_seconds / TIME_GRID_STEP_SECONDS)))
    n_y_small = max(1, int(round((2 * DISPLAY_Y_LIMIT_MV) / DISPLAY_Y_GRID_STEP_MV)))

    for lead_index, lead_name in enumerate(LEAD_ORDER):
        row, col = divmod(lead_index, 3)
        panel_x = margin_x + col * (panel_w + gap_x)
        panel_y = top + row * (panel_h + gap_y)

        x0, x1 = panel_x + 8, panel_x + panel_w - 8
        y0, y1 = panel_y + 30, panel_y + panel_h - 8
        plot_h = y1 - y0

        draw.rounded_rectangle(
            (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
            radius=5,
            outline=(210, 210, 210),
            width=1,
            fill=(255, 255, 255),
        )
        draw.text(
            (panel_x + 10, panel_y + 5),
            lead_name,
            fill=(25, 25, 25),
            font=lead_font,
        )

        # Horizontal grid: 0.04 s minor intervals; every fifth line is 0.20 s.
        for k in range(n_x_small + 1):
            x = int(round(x0 + (k / n_x_small) * (x1 - x0)))
            major = (k % 5 == 0)
            color = (225, 170, 175) if major else (244, 220, 222)
            draw.line((x, y0, x, y1), fill=color, width=1)

        # Vertical grid: 0.2 mV minor intervals; every fifth line is 1.0 mV.
        for k in range(n_y_small + 1):
            y = int(round(y1 - (k / n_y_small) * plot_h))
            major = (k % 5 == 0)
            color = (225, 170, 175) if major else (244, 220, 222)
            draw.line((x0, y, x1, y), fill=color, width=1)

        wave = signal_mV[:, lead_index]
        valid = np.isfinite(wave)
        # Use sample time (n/fs), so the final sample remains one sample before 1.2 s.
        sample_times = np.arange(len(wave), dtype=np.float64) / sampling_rate_hz
        x_values = x0 + (sample_times / duration_seconds) * (x1 - x0)
        safe_wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
        clipped_wave = np.clip(safe_wave, -DISPLAY_Y_LIMIT_MV, DISPLAY_Y_LIMIT_MV)
        y_values = y1 - (
            (clipped_wave + DISPLAY_Y_LIMIT_MV) / (2 * DISPLAY_Y_LIMIT_MV)
        ) * plot_h
        draw_waveform_segments(draw, x_values, y_values, valid)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return clipped_fraction, max_abs_mV


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert PTB-XL+ 12SL median-beat WFDB records to 12-lead PNG images."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"Extracted 12SL directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--max-records", type=int, default=10,
        help="Convert only the first N header files (default: 10).",
    )
    selection.add_argument(
        "--all", action="store_true",
        help="Convert every discovered 12SL record.",
    )
    parser.add_argument(
        "--features-csv", type=Path, default=None,
        help="Optional PTB-XL+ 12sl_features.csv for reproducible amplitude-scale validation.",
    )
    parser.add_argument(
        "--display-y-limit-mv", type=float, default=DISPLAY_Y_LIMIT_MV,
        help=f"Fixed positive/negative display limit in mV (default: +/-{DISPLAY_Y_LIMIT_MV}).",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    y_limit_mV = float(args.display_y_limit_mv)

    if not input_dir.is_dir():
        print(
            f"ERROR: input directory does not exist: {input_dir}\n"
            "Extract the 12sl.zip archive and point --input-dir to the extracted 12sl folder.",
            file=sys.stderr,
        )
        return 2
    if not np.isfinite(y_limit_mV) or y_limit_mV <= 0:
        print("ERROR: --display-y-limit-mv must be positive and finite", file=sys.stderr)
        return 2
    if args.max_records is not None and args.max_records < 1:
        print("ERROR: --max-records must be at least 1", file=sys.stderr)
        return 2

    # Set the module-level display range to the CLI value used by the renderer.
    global DISPLAY_Y_LIMIT_MV
    DISPLAY_Y_LIMIT_MV = y_limit_mV

    try:
        feature_reference = load_feature_reference(args.features_csv)
    except Exception as exc:
        print(f"ERROR loading feature reference: {exc}", file=sys.stderr)
        return 2

    headers = sorted(input_dir.rglob("*.hea"))
    if not headers:
        print(f"ERROR: no .hea files found under {input_dir}", file=sys.stderr)
        return 2
    if not args.all:
        headers = headers[: args.max_records]

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, str]] = []
    lead_quality_rows: list[dict[str, Any]] = []
    calibration_accumulator: dict[tuple[str, str], dict[str, list[float]]] = {}
    seen_ecg_ids: set[int] = set()
    matched_feature_records = 0

    for header_path in tqdm(headers, desc="Converting 12SL median beats"):
        try:
            ecg_id, signal_mV, header_scaled, details, lead_rows = decode_record(header_path)
            if ecg_id in seen_ecg_ids:
                raise ValueError(f"Duplicate ECG ID encountered: {ecg_id}")
            seen_ecg_ids.add(ecg_id)

            feature_row = None
            if feature_reference is not None and ecg_id in feature_reference.index:
                feature_row = feature_reference.loc[ecg_id]
                matched_feature_records += 1
                add_calibration_comparisons(
                    calibration_accumulator,
                    header_scaled_signal=header_scaled,
                    lead_names=details["lead_names"],
                    feature_row=feature_row,
                )

            image_relative = Path("images") / f"{ecg_id:05d}.png"
            clipped_fraction, max_abs_mV = render_12lead_image(
                signal_mV=signal_mV,
                lead_names=details["lead_names"],
                ecg_id=ecg_id,
                sampling_rate_hz=details["sampling_rate_hz"],
                output_path=output_dir / image_relative,
            )

            technical_reasons: list[str] = []
            advisory_reasons: list[str] = []
            if details["wfdb_invalid_samples"] > 0:
                technical_reasons.append("standard_wfdb_invalid_samples_present")
            if clipped_fraction > 0:
                technical_reasons.append("one_or_more_samples_clipped_in_png")
            if details["max_baseline_run_samples"] > LONG_BASELINE_RUN_REVIEW_SAMPLES:
                advisory_reasons.append("long_constant_baseline_run_possible_padding")

            for lead_row in lead_rows:
                lead_quality_rows.append({"ecg_id": ecg_id, **lead_row})

            manifest_rows.append(
                {
                    "ecg_id": ecg_id,
                    "image_path": image_relative.as_posix(),
                    "source": "PTB-XL+_12SL_median_beats",
                    "sampling_rate_hz": details["sampling_rate_hz"],
                    "samples_per_lead": details["samples_per_lead"],
                    "duration_seconds": details["duration_seconds"],
                    "n_leads": details["n_leads"],
                    "lead_names": "|".join(details["lead_names"]),
                    "header_units_token": details["header_units"],
                    "wfdb_invalid_samples": details["wfdb_invalid_samples"],
                    "wfdb_invalid_fraction": round(details["wfdb_invalid_fraction"], 8),
                    "max_invalid_gap_samples": details["max_invalid_gap_samples"],
                    "max_baseline_run_samples": details["max_baseline_run_samples"],
                    "n_leads_with_long_baseline_run": details["n_leads_with_long_baseline_run"],
                    "near_int32_rail_samples": details["near_int32_rail_samples"],
                    "near_rail_nonbaseline_samples": details["near_rail_nonbaseline_samples"],
                    "display_divisor": DISPLAY_DIVISOR,
                    "display_units": "mV_empirically_cross_checked_against_12SL_features",
                    "display_y_limit_mV": y_limit_mV,
                    "display_max_abs_mV": max_abs_mV,
                    "clipped_fraction": round(clipped_fraction, 8),
                    "technical_review_flag": "review" if technical_reasons else "ok",
                    "technical_review_reasons": ";".join(technical_reasons),
                    "advisory_flag": bool(advisory_reasons),
                    "advisory_reasons": ";".join(advisory_reasons),
                    # Label and split columns are intentionally blank at this stage.
                    "target": "",
                    "target_id": "",
                    "patient_id": "",
                    "strat_fold": "",
                }
            )

        except Exception as exc:
            try:
                relative_header = header_path.relative_to(input_dir).as_posix()
            except ValueError:
                relative_header = str(header_path)
            failure_rows.append(
                {
                    "header_path": relative_header,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    manifest_columns = [
        "ecg_id", "image_path", "source", "sampling_rate_hz", "samples_per_lead",
        "duration_seconds", "n_leads", "lead_names", "header_units_token",
        "wfdb_invalid_samples", "wfdb_invalid_fraction", "max_invalid_gap_samples",
        "max_baseline_run_samples", "n_leads_with_long_baseline_run",
        "near_int32_rail_samples", "near_rail_nonbaseline_samples", "display_divisor",
        "display_units", "display_y_limit_mV", "display_max_abs_mV", "clipped_fraction",
        "technical_review_flag", "technical_review_reasons", "advisory_flag",
        "advisory_reasons", "target", "target_id", "patient_id", "strat_fold",
    ]
    manifest = pd.DataFrame(manifest_rows, columns=manifest_columns)
    if not manifest.empty:
        manifest = manifest.sort_values("ecg_id")
    manifest.to_csv(output_dir / "manifest_unlabelled.csv", index=False)

    failures = pd.DataFrame(
        failure_rows, columns=["header_path", "error_type", "error"]
    )
    failures.to_csv(output_dir / "conversion_failures.csv", index=False)

    lead_columns = [
        "ecg_id", "lead", "header_units_token", "adc_gain", "baseline_digital",
        "invalid_samples", "invalid_fraction", "max_invalid_gap_samples",
        "baseline_equal_samples", "max_baseline_run_samples", "near_int32_rail_samples",
        "near_rail_nonbaseline_samples", "header_value_min", "header_value_max",
    ]
    lead_quality = pd.DataFrame(lead_quality_rows, columns=lead_columns)
    lead_quality.to_csv(output_dir / "lead_quality.csv", index=False)

    calibration_frame, calibration_summary = build_calibration_summary(calibration_accumulator)
    if feature_reference is not None:
        calibration_frame.to_csv(output_dir / "calibration_validation_summary.csv", index=False)
        (output_dir / "calibration_validation.json").write_text(
            json.dumps(
                {
                    "features_csv": str(args.features_csv),
                    "matched_feature_records": matched_feature_records,
                    **calibration_summary,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    header_units_found = sorted({row["header_units_token"] for row in manifest_rows})
    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "headers_selected": len(headers),
        "images_created": len(manifest),
        "conversion_failures": len(failures),
        "image_size_pixels": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "image_layout": "4 rows x 3 columns; standard 12-lead order",
        "expected_sampling_rate_hz": EXPECTED_FS_HZ,
        "expected_samples_per_lead": EXPECTED_SAMPLES,
        "expected_duration_seconds": EXPECTED_DURATION_S,
        "wfdb_format": 32,
        "raw_dtype": RAW_DTYPE.str,
        "wfdb_invalid_value": WFDB_INVALID_SAMPLE,
        "invalid_policy": "exact INT32_MIN only",
        "interpolation": "none; invalid samples drawn as visible gaps",
        "baseline_policy": "preserve baseline-valued samples; long runs reported as advisory",
        "header_units_found": header_units_found,
        "display_divisor": DISPLAY_DIVISOR,
        "display_units": "mV scale empirically cross-checked against 12SL R/Q/S amplitude features",
        "display_y_limit_mV": y_limit_mV,
        "grid_spacing": {
            "horizontal_seconds": TIME_GRID_STEP_SECONDS,
            "vertical_mV": DISPLAY_Y_GRID_STEP_MV,
            "note": "Consistent visualization grid; not a certified physical pixel/mm ECG-paper scale",
        },
        "calibration_validation": calibration_summary,
        "features_csv_used": str(args.features_csv) if args.features_csv else None,
        "matched_feature_records": matched_feature_records,
        "labels_attached": False,
        "patient_splits_attached": False,
        "quality_warning": "Technical/advisory flags are not clinical ECG-quality diagnoses.",
    }
    (output_dir / "conversion_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\nConversion summary")
    print(json.dumps(config, indent=2))
    print(f"\nManifest: {output_dir / 'manifest_unlabelled.csv'}")
    print(f"Failures: {output_dir / 'conversion_failures.csv'}")
    print(f"Lead quality: {output_dir / 'lead_quality.csv'}")
    if feature_reference is not None:
        print(f"Calibration summary: {output_dir / 'calibration_validation.json'}")

    return 1 if len(failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
