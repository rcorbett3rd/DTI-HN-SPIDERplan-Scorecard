from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath


@dataclass
class DoseGeometry:
    dose: np.ndarray
    row_spacing: float
    col_spacing: float
    slice_spacing: float
    image_position: np.ndarray
    row_dir: np.ndarray
    col_dir: np.ndarray
    slice_dir: np.ndarray
    z_offsets: np.ndarray
    dose_units: str
    dose_type: str


def _as_float_array(value, default) -> np.ndarray:
    try:
        return np.asarray([float(v) for v in value], dtype=float)
    except Exception:
        return np.asarray(default, dtype=float)


def get_dose_geometry(rtdose: Any) -> DoseGeometry:
    arr = np.asarray(rtdose.pixel_array, dtype=float)
    scaling = float(getattr(rtdose, "DoseGridScaling", 1.0) or 1.0)
    dose = arr * scaling
    if dose.ndim == 2:
        dose = dose[np.newaxis, :, :]

    spacing = _as_float_array(getattr(rtdose, "PixelSpacing", [1.0, 1.0]), [1.0, 1.0])
    row_spacing = float(spacing[0])
    col_spacing = float(spacing[1])
    ipp = _as_float_array(getattr(rtdose, "ImagePositionPatient", [0, 0, 0]), [0, 0, 0])
    iop = _as_float_array(getattr(rtdose, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), [1, 0, 0, 0, 1, 0])
    row_dir = iop[:3]
    col_dir = iop[3:]
    slice_dir = np.cross(row_dir, col_dir)
    norm = np.linalg.norm(slice_dir)
    if norm > 0:
        slice_dir = slice_dir / norm

    offsets = getattr(rtdose, "GridFrameOffsetVector", None)
    if offsets is not None:
        z_offsets = np.asarray([float(x) for x in offsets], dtype=float)
    else:
        thickness = float(getattr(rtdose, "SliceThickness", 1.0) or 1.0)
        z_offsets = np.arange(dose.shape[0], dtype=float) * thickness
    if len(z_offsets) > 1:
        slice_spacing = float(np.median(np.abs(np.diff(z_offsets))))
    else:
        slice_spacing = float(getattr(rtdose, "SliceThickness", 1.0) or 1.0)

    dose_units = str(getattr(rtdose, "DoseUnits", "GY") or "GY").upper()
    dose_type = str(getattr(rtdose, "DoseType", "") or "").upper()

    # RT Dose is usually stored in Gy. If a non-standard export stores cGy-like
    # physical values, normalize to Gy by magnitude. This avoids false coverage
    # failure from unit mismatch while preserving ordinary Gy exports.
    finite = dose[np.isfinite(dose)]
    if dose_units in {"CGY", "CENTIGRAY"}:
        dose = dose / 100.0
        dose_units = "GY"
    elif dose_units == "GY" and finite.size and np.nanpercentile(finite, 99.9) > 250:
        dose = dose / 100.0
        dose_units = "GY_ASSUMED_FROM_CGY_VALUES"

    return DoseGeometry(dose, row_spacing, col_spacing, slice_spacing, ipp, row_dir, col_dir, slice_dir, z_offsets, dose_units, dose_type)


def _roi_name_map(rtstruct: Any) -> dict[int, str]:
    out = {}
    for roi in getattr(rtstruct, "StructureSetROISequence", []) or []:
        try:
            out[int(getattr(roi, "ROINumber"))] = str(getattr(roi, "ROIName", "Unnamed") or "Unnamed")
        except Exception:
            pass
    return out




def _is_body_or_external_name(name: str) -> bool:
    n = str(name).strip().lower()
    return n == "body" or n == "external" or n.startswith("external")


def _is_non_scored_helper_name(name: str) -> bool:
    """Structures sampled only when specifically needed elsewhere. BODY/External is handled by global_hotspot_analysis."""
    n = str(name).strip().lower()
    tokens = [t for t in __import__('re').split(r"[^a-z0-9]+", n) if t]
    is_ln = "ln" in tokens or n.startswith("ln") or n.endswith("ln")
    return n.startswith("z") or is_ln or _is_body_or_external_name(name)

def _patient_to_voxel_variant(points: np.ndarray, geom: DoseGeometry, rowcol_mode: str, frame_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map patient points to dose-grid frames/rows/columns.

    This app evaluates multiple safe geometry interpretations. Most
    Eclipse exports follow the standard DICOM mapping, but some prototype
    exports or anonymized datasets can appear transposed or use absolute GFOV
    values. Trying variants prevents false D95=0 failures without crashing.
    """
    delta = points - geom.image_position

    if rowcol_mode == "standard":
        # DICOM: P = IPP + col*column_spacing*row_dir + row*row_spacing*col_dir
        cols = delta @ geom.row_dir / geom.col_spacing
        rows = delta @ geom.col_dir / geom.row_spacing
    elif rowcol_mode == "swapped":
        rows = delta @ geom.row_dir / geom.row_spacing
        cols = delta @ geom.col_dir / geom.col_spacing
    else:
        raise ValueError(f"Unknown row/column mapping mode: {rowcol_mode}")

    if frame_mode == "relative":
        slice_positions = delta @ geom.slice_dir
    elif frame_mode == "absolute":
        slice_positions = points @ geom.slice_dir
    else:
        raise ValueError(f"Unknown frame mapping mode: {frame_mode}")

    if len(geom.z_offsets) > 0:
        frames = np.array([int(np.argmin(np.abs(geom.z_offsets - z))) for z in slice_positions])
    else:
        frames = np.zeros(len(points), dtype=int)
    return frames, rows, cols


def _dose_values_for_roi_variant(
    rtstruct: Any,
    roi_number: int,
    geom: DoseGeometry,
    rowcol_mode: str,
    frame_mode: str,
    max_voxels: int = 300000,
) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    values: list[np.ndarray] = []
    rows_n = geom.dose.shape[1]
    cols_n = geom.dose.shape[2]

    target = None
    for rc in getattr(rtstruct, "ROIContourSequence", []) or []:
        try:
            if int(getattr(rc, "ReferencedROINumber")) == int(roi_number):
                target = rc
                break
        except Exception:
            continue
    if target is None:
        return np.array([], dtype=float), ["No ROIContourSequence contour found"]

    contours = getattr(target, "ContourSequence", []) or []
    if not contours:
        return np.array([], dtype=float), ["Structure has no contour sequence"]

    for contour in contours:
        try:
            data = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            if data.shape[0] < 3:
                continue
            frames, rr, cc = _patient_to_voxel_variant(data, geom, rowcol_mode, frame_mode)
            frame = int(np.median(frames))
            if frame < 0 or frame >= geom.dose.shape[0]:
                continue
            rmin = max(int(np.floor(np.nanmin(rr))) - 1, 0)
            rmax = min(int(np.ceil(np.nanmax(rr))) + 1, rows_n - 1)
            cmin = max(int(np.floor(np.nanmin(cc))) - 1, 0)
            cmax = min(int(np.ceil(np.nanmax(cc))) + 1, cols_n - 1)
            if rmax <= rmin or cmax <= cmin:
                continue

            rgrid, cgrid = np.mgrid[rmin:rmax + 1, cmin:cmax + 1]
            pix = np.vstack((rgrid.ravel(), cgrid.ravel())).T
            poly = np.vstack((rr, cc)).T
            mask = MplPath(poly).contains_points(pix)
            if np.any(mask):
                vals = geom.dose[frame, rgrid.ravel()[mask], cgrid.ravel()[mask]]
                values.append(vals[np.isfinite(vals)])
        except Exception as exc:
            warnings.append(f"Skipped one contour: {exc}")
            continue

    if not values:
        return np.array([], dtype=float), warnings + ["No dose voxels sampled for this structure"]
    all_values = np.concatenate(values)
    if all_values.size > max_voxels:
        rng = np.random.default_rng(42)
        all_values = rng.choice(all_values, size=max_voxels, replace=False)
        warnings.append("Voxel sample was downsampled for app performance")
    return all_values, warnings


def _choose_best_variant(rtstruct: Any, roi_number: int, geom: DoseGeometry, assigned_rx: float | None) -> tuple[np.ndarray, list[str], str]:
    variants = [("standard", "relative"), ("standard", "absolute"), ("swapped", "relative"), ("swapped", "absolute")]
    results = []
    for rowcol_mode, frame_mode in variants:
        vals, warnings = _dose_values_for_roi_variant(rtstruct, roi_number, geom, rowcol_mode, frame_mode)
        if vals.size == 0:
            score = -1.0
        elif assigned_rx and assigned_rx > 0:
            # For scored targets, select the geometry that actually samples the
            # prescribed dose cloud instead of low-dose/zero background.
            v95 = float(np.mean(vals >= 0.95 * assigned_rx) * 100)
            d95 = float(np.percentile(vals, 5))
            nonzero = float(np.mean(vals > 0.01) * 100)
            score = v95 * 1000.0 + (d95 / assigned_rx * 100.0) + nonzero * 0.01
        else:
            # For OAR/other structures, use the variant with the most sampled,
            # nonzero, plausible dose information.
            nonzero = float(np.mean(vals > 0.01) * 100)
            score = nonzero * 100.0 + float(np.nanmean(vals)) + np.log1p(vals.size)
        results.append((score, vals, warnings, f"{rowcol_mode}/{frame_mode}"))

    results.sort(key=lambda x: x[0], reverse=True)
    best_score, best_vals, best_warnings, best_name = results[0]
    if best_vals.size == 0:
        # Preserve useful warnings from all attempts.
        all_warnings: list[str] = []
        for _, _, warnings, name in results:
            all_warnings.extend([f"{name}: {w}" for w in warnings[:1]])
        return best_vals, all_warnings or ["No dose voxels sampled for this structure"], best_name
    return best_vals, best_warnings, best_name


def _voxel_indices_for_roi_variant(
    rtstruct: Any,
    roi_number: int,
    geom: DoseGeometry,
    rowcol_mode: str,
    frame_mode: str,
    max_voxels: int = 900000,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return sampled dose values and integer frame/row/column indices for an ROI."""
    warnings: list[str] = []
    values: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    rows_n = geom.dose.shape[1]
    cols_n = geom.dose.shape[2]

    target = None
    for rc in getattr(rtstruct, "ROIContourSequence", []) or []:
        try:
            if int(getattr(rc, "ReferencedROINumber")) == int(roi_number):
                target = rc
                break
        except Exception:
            continue
    if target is None:
        return np.array([], dtype=float), np.empty((0, 3), dtype=int), ["No ROIContourSequence contour found"]

    contours = getattr(target, "ContourSequence", []) or []
    if not contours:
        return np.array([], dtype=float), np.empty((0, 3), dtype=int), ["Structure has no contour sequence"]

    for contour in contours:
        try:
            data = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            if data.shape[0] < 3:
                continue
            frames, rr, cc = _patient_to_voxel_variant(data, geom, rowcol_mode, frame_mode)
            frame = int(np.median(frames))
            if frame < 0 or frame >= geom.dose.shape[0]:
                continue
            rmin = max(int(np.floor(np.nanmin(rr))) - 1, 0)
            rmax = min(int(np.ceil(np.nanmax(rr))) + 1, rows_n - 1)
            cmin = max(int(np.floor(np.nanmin(cc))) - 1, 0)
            cmax = min(int(np.ceil(np.nanmax(cc))) + 1, cols_n - 1)
            if rmax <= rmin or cmax <= cmin:
                continue
            rgrid, cgrid = np.mgrid[rmin:rmax + 1, cmin:cmax + 1]
            pix = np.vstack((rgrid.ravel(), cgrid.ravel())).T
            poly = np.vstack((rr, cc)).T
            mask = MplPath(poly).contains_points(pix)
            if np.any(mask):
                rr_idx = rgrid.ravel()[mask].astype(int)
                cc_idx = cgrid.ravel()[mask].astype(int)
                ff_idx = np.full(rr_idx.shape, frame, dtype=int)
                vals = geom.dose[ff_idx, rr_idx, cc_idx]
                finite = np.isfinite(vals)
                values.append(vals[finite])
                indices.append(np.vstack((ff_idx[finite], rr_idx[finite], cc_idx[finite])).T)
        except Exception as exc:
            warnings.append(f"Skipped one contour: {exc}")
            continue

    if not values:
        return np.array([], dtype=float), np.empty((0, 3), dtype=int), warnings + ["No dose voxels sampled for this structure"]
    all_values = np.concatenate(values)
    all_indices = np.vstack(indices)
    if all_values.size > max_voxels:
        rng = np.random.default_rng(42)
        pick = rng.choice(np.arange(all_values.size), size=max_voxels, replace=False)
        all_values = all_values[pick]
        all_indices = all_indices[pick]
        warnings.append("Voxel sample was downsampled for app performance")
    return all_values, all_indices, warnings


def global_hotspot_analysis(
    rtstruct: Any,
    rtdose: Any,
    rx_map: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Evaluate global max dose only in BODY/External and check whether it is inside a PTV.

    Tolerance is normalized to the highest assigned target Rx:
    preferred <=107%, marginal >107-110%, fail >110% or outside PTV.
    """
    warnings: list[str] = []
    try:
        geom = get_dose_geometry(rtdose)
    except Exception as exc:
        return pd.DataFrame(), [f"Could not read RT Dose for global hotspot analysis: {exc}"]

    roi_names = _roi_name_map(rtstruct)
    body_items = [(num, name) for num, name in roi_names.items() if str(name).strip().lower() == "body" or str(name).strip().lower().startswith("external")]
    if not body_items:
        return pd.DataFrame([{
            "structure": "BODY/External",
            "category": "Global Max Hotspot Review",
            "scoring_role": "Global max hotspot",
            "score": 60.0,
            "grade": "D",
            "notes": "No BODY or External contour found for global hotspot review",
        }]), warnings

    rx_values = [float(v) for v in (rx_map or {}).values() if v and float(v) > 0]
    highest_rx = max(rx_values) if rx_values else None

    # Use the same geometry variant as targets when possible by choosing the variant with the most nonzero BODY samples.
    variants = [("standard", "relative"), ("standard", "absolute"), ("swapped", "relative"), ("swapped", "absolute")]
    best = None
    for roi_number, body_name in body_items:
        for rowcol_mode, frame_mode in variants:
            vals, idx, w = _voxel_indices_for_roi_variant(rtstruct, roi_number, geom, rowcol_mode, frame_mode)
            if vals.size == 0:
                continue
            nonzero = float(np.mean(vals > 0.01) * 100)
            score = nonzero * 100.0 + float(np.nanmax(vals)) + np.log1p(vals.size)
            candidate = (score, vals, idx, body_name, f"{rowcol_mode}/{frame_mode}")
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        return pd.DataFrame([{
            "structure": body_items[0][1],
            "category": "Global Max Hotspot Review",
            "scoring_role": "Global max hotspot",
            "score": 60.0,
            "grade": "D",
            "notes": "No sampled dose in BODY/External for global hotspot review",
        }]), warnings

    _, body_vals, body_idx, body_name, variant = best
    max_pos = int(np.nanargmax(body_vals))
    global_max = float(body_vals[max_pos])
    max_idx = tuple(int(x) for x in body_idx[max_pos])

    ptv_items = []
    # Only the highest-dose PTV level is acceptable for the global maximum hotspot.
    # Lower-dose PTVs and *_eval/PTV optimization helper contours are not counted
    # as acceptable global-hotspot locations.
    for num, name in roi_names.items():
        n = str(name).strip().lower()
        tokens = [t for t in __import__('re').split(r"[^a-z0-9]+", n) if t]
        is_ln = "ln" in tokens or n.startswith("ln") or n.endswith("ln")
        if n.startswith("z") or n.endswith("_eval") or n.endswith("opti") or is_ln:
            continue
        if "ptv" not in n:
            continue
        this_rx = None
        if rx_map:
            this_rx = rx_map.get(str(name))
        try:
            if highest_rx is not None and this_rx is not None and abs(float(this_rx) - float(highest_rx)) < 0.05:
                ptv_items.append((num, name))
        except Exception:
            continue

    inside_ptv = False
    containing_ptvs: list[str] = []
    max_key = f"{max_idx[0]},{max_idx[1]},{max_idx[2]}"
    for roi_number, ptv_name in ptv_items:
        vals, idx, w = _voxel_indices_for_roi_variant(rtstruct, roi_number, geom, *variant.split("/"))
        if idx.size == 0:
            continue
        idx_keys = {f"{int(f)},{int(r)},{int(c)}" for f, r, c in idx}
        if max_key in idx_keys:
            inside_ptv = True
            containing_ptvs.append(str(ptv_name))

    percent_rx = round(global_max / highest_rx * 100, 1) if highest_rx else np.nan
    score = 100.0
    notes = []
    if not inside_ptv:
        score -= 45
        notes.append("Global max hotspot is outside the highest-dose PTV")
    else:
        notes.append("Global max hotspot is within highest-dose PTV: " + ", ".join(containing_ptvs[:3]))
    if highest_rx:
        if percent_rx >= 110:
            score -= 40
            notes.append("Global max is fail range: ≥110% of highest Rx")
        elif percent_rx > 107:
            score -= 15
            notes.append("Global max is marginal range: >107–<110% of highest Rx")
        else:
            notes.append("Global max is preferred range: ≤107% of highest Rx")
    else:
        score -= 10
        notes.append("No assigned target Rx available for global max %Rx normalization")

    score = max(score, 0.0)
    grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"
    row = {
        "structure": str(body_name),
        "category": "Global Max Hotspot Review",
        "scoring_role": "BODY/External global max + PTV location",
        "assigned_rx_gy": round(float(highest_rx), 3) if highest_rx else np.nan,
        "Dmax_Gy": round(global_max, 3),
        "GlobalMax_%Rx": percent_rx,
        "hotspot_inside_ptv": inside_ptv,
        "containing_ptv": ", ".join(containing_ptvs),
        "score": round(score, 1),
        "grade": grade,
        "notes": "; ".join(notes),
    }
    warnings.append(f"Global max hotspot from {body_name} using {variant}: {global_max:.3f} Gy at voxel {max_idx}; inside_ptv={inside_ptv}")
    return pd.DataFrame([row]), warnings

def calculate_dvh_metrics(
    rtstruct: Any,
    rtdose: Any,
    rx_dose_gy: float | None = None,
    structure_limit: int | None = None,
    rx_map: dict[str, float] | None = None,
    priority_structures: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    app_warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    try:
        geom = get_dose_geometry(rtdose)
    except Exception as exc:
        return pd.DataFrame(), [f"Could not read RT Dose pixel data: {exc}"]

    try:
        rs_for = str(getattr(rtstruct, "FrameOfReferenceUID", ""))
        rd_for = str(getattr(rtdose, "FrameOfReferenceUID", ""))
        if rs_for and rd_for and rs_for != rd_for:
            app_warnings.append("RTSTRUCT and RTDOSE FrameOfReferenceUID differ. Dose/contour mapping may be invalid.")
    except Exception:
        pass

    roi_names = _roi_name_map(rtstruct)
    roi_items = list(roi_names.items())

    if priority_structures:
        priority_keys = {str(x).strip().lower() for x in priority_structures}
        priority = [(num, name) for num, name in roi_items if str(name).strip().lower() in priority_keys]
        remainder = [(num, name) for num, name in roi_items if str(name).strip().lower() not in priority_keys]
        roi_items = priority + remainder

    if structure_limit:
        roi_items = roi_items[:structure_limit]

    voxel_volume_cc = (geom.row_spacing * geom.col_spacing * geom.slice_spacing) / 1000.0

    dose_min = float(np.nanmin(geom.dose)) if geom.dose.size else float("nan")
    dose_max = float(np.nanmax(geom.dose)) if geom.dose.size else float("nan")
    app_warnings.append(f"Dose grid read: shape={geom.dose.shape}, units={geom.dose_units}, min={dose_min:.3f} Gy, max={dose_max:.3f} Gy")

    for roi_number, name in roi_items:
        try:
            if _is_non_scored_helper_name(name):
                continue
            assigned_rx = None
            if rx_map is not None:
                # Critical: when per-structure Rx values are supplied, do NOT fall back
                # to one global plan Rx. Lower-dose PTV/CTV levels must be evaluated
                # against their own assigned dose, not the highest prescription.
                assigned_rx = rx_map.get(str(name))
            elif rx_dose_gy and rx_dose_gy > 0:
                assigned_rx = rx_dose_gy

            vals, warnings, variant = _choose_best_variant(rtstruct, roi_number, geom, assigned_rx)
            if warnings:
                app_warnings.extend([f"{name}: {w}" for w in warnings[:2]])
            if assigned_rx and vals.size > 0:
                v95_preview = float(np.mean(vals >= 0.95 * assigned_rx) * 100)
                d95_preview = float(np.percentile(vals, 5))
                app_warnings.append(f"{name}: selected geometry {variant}; sampled_voxels={vals.size}; D95={d95_preview:.2f} Gy; V95Rx={v95_preview:.1f}%")
            elif vals.size > 0:
                app_warnings.append(f"{name}: selected geometry {variant}; sampled_voxels={vals.size}")

            if vals.size == 0:
                rows.append({"structure": name, "status": "No sampled dose", "sampled_voxels": 0})
                continue
            dmax = float(np.max(vals))
            dmean = float(np.mean(vals))
            dmin = float(np.min(vals))
            d95 = float(np.percentile(vals, 5))
            d98 = float(np.percentile(vals, 2))
            d2 = float(np.percentile(vals, 98))
            d5 = float(np.percentile(vals, 95))
            sorted_desc = np.sort(vals)[::-1]
            voxels_003cc = max(int(np.ceil(0.03 / voxel_volume_cc)), 1) if voxel_volume_cc > 0 else 1
            voxels_003cc = min(voxels_003cc, sorted_desc.size)
            d003cc = float(sorted_desc[voxels_003cc - 1])
            row = {
                "structure": name,
                "status": "Calculated",
                "geometry_mode": variant,
                "sampled_voxels": int(vals.size),
                "approx_volume_cc": round(float(vals.size * voxel_volume_cc), 2),
                "Dmin_Gy": round(dmin, 3),
                "Dmean_Gy": round(dmean, 3),
                "Dmax_Gy": round(dmax, 3),
                "D0.03cc_Gy": round(d003cc, 3),
                "D98_Gy": round(d98, 3),
                "D95_Gy": round(d95, 3),
                "D5_Gy": round(d5, 3),
                "D2_Gy": round(d2, 3),
                "V30Gy_%": round(float(np.mean(vals >= 30.0) * 100), 1),
            }
            if assigned_rx and assigned_rx > 0:
                row["assigned_rx_gy"] = round(float(assigned_rx), 3)
                row["D95_%Rx"] = round(d95 / assigned_rx * 100, 1)
                row["Dmin_%Rx"] = round(dmin / assigned_rx * 100, 1)
                row["V100Rx_%"] = round(float(np.mean(vals >= 1.00 * assigned_rx) * 100), 1)
                row["D0.03cc_%Rx"] = round(d003cc / assigned_rx * 100, 1)
                row["D2_%Rx"] = round(d2 / assigned_rx * 100, 1)
                row["V95Rx_%"] = round(float(np.mean(vals >= 0.95 * assigned_rx) * 100), 1)
                row["V105Rx_%"] = round(float(np.mean(vals >= 1.05 * assigned_rx) * 100), 1)
            else:
                row["assigned_rx_gy"] = np.nan
            rows.append(row)
        except Exception as exc:
            rows.append({"structure": name, "status": f"Skipped: {exc}", "sampled_voxels": 0})
            continue

    return pd.DataFrame(rows), app_warnings


def dvh_note() -> str:
    return (
        "Prototype DVH metrics are contour-sampled from RTSTRUCT polygons onto the RT Dose grid. "
        "This build uses a guarded multi-geometry mapper to reduce false target-coverage failures when RT Dose row/column or frame offsets are interpreted differently. "
        "Validate all outputs against your clinical TPS before clinical use."
    )
