from __future__ import annotations

from pathlib import Path
from typing import Any

import pydicom
from pydicom.errors import InvalidDicomError


def save_uploaded_files(uploaded_files, upload_dir: str | Path = "uploaded_dicoms") -> list[Path]:
    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for uploaded_file in uploaded_files:
        safe_name = Path(uploaded_file.name).name
        destination = upload_path / safe_name
        with open(destination, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(destination)
    return saved_paths


def load_dicoms(paths: list[str | Path]) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        try:
            ds = pydicom.dcmread(str(p), force=True, stop_before_pixels=False)
            modality = str(getattr(ds, "Modality", "UNKNOWN") or "UNKNOWN").upper()
            loaded.append({"path": p, "dataset": ds, "modality": modality})
        except (InvalidDicomError, Exception):
            continue
    return loaded


def classify_rt_files(dicoms: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"RTPLAN": None, "RTSTRUCT": None, "RTDOSE": None, "OTHER": []}
    for item in dicoms:
        ds = item.get("dataset")
        modality = str(item.get("modality", "UNKNOWN")).upper()
        if modality == "RTPLAN" and result["RTPLAN"] is None:
            result["RTPLAN"] = ds
        elif modality == "RTSTRUCT" and result["RTSTRUCT"] is None:
            result["RTSTRUCT"] = ds
        elif modality == "RTDOSE" and result["RTDOSE"] is None:
            result["RTDOSE"] = ds
        else:
            result["OTHER"].append(item)
    return result


def _safe_get(ds: Any, attr: str, default: str = "") -> str:
    value = getattr(ds, attr, default)
    if value is None:
        return default
    return str(value)


def extract_plan_summary(rtplan: Any | None) -> dict[str, Any]:
    if rtplan is None:
        return {}
    beams = getattr(rtplan, "BeamSequence", []) or []
    fraction_groups = getattr(rtplan, "FractionGroupSequence", []) or []
    prescriptions = getattr(rtplan, "DoseReferenceSequence", []) or []
    total_fractions = ""
    if fraction_groups:
        total_fractions = _safe_get(fraction_groups[0], "NumberOfFractionsPlanned", "")
    dose_refs = []
    for ref in prescriptions:
        dose = getattr(ref, "TargetPrescriptionDose", None)
        if dose is not None:
            try:
                dose_refs.append(float(dose))
            except Exception:
                pass
    rx = max(dose_refs) if dose_refs else None
    return {
        "plan_label": _safe_get(rtplan, "RTPlanLabel", "Unknown"),
        "plan_name": _safe_get(rtplan, "RTPlanName", ""),
        "plan_date": _safe_get(rtplan, "RTPlanDate", ""),
        "plan_time": _safe_get(rtplan, "RTPlanTime", ""),
        "beam_count": len(beams),
        "fraction_groups": len(fraction_groups),
        "fractions_planned": total_fractions,
        "highest_prescription_Gy": rx if rx is not None else "Not found",
    }


def get_prescription_dose_gy(rtplan: Any | None) -> float | None:
    if rtplan is None:
        return None
    doses: list[float] = []
    for ref in getattr(rtplan, "DoseReferenceSequence", []) or []:
        dose = getattr(ref, "TargetPrescriptionDose", None)
        if dose is not None:
            try:
                doses.append(float(dose))
            except Exception:
                pass
    return max(doses) if doses else None


def extract_structures(rtstruct: Any | None) -> list[dict[str, Any]]:
    if rtstruct is None:
        return []
    roi_sequence = getattr(rtstruct, "StructureSetROISequence", []) or []
    observations = getattr(rtstruct, "RTROIObservationsSequence", []) or []
    obs_by_number = {}
    for obs in observations:
        roi_num = getattr(obs, "ReferencedROINumber", None)
        if roi_num is not None:
            try:
                obs_by_number[int(roi_num)] = obs
            except Exception:
                pass
    structures: list[dict[str, Any]] = []
    for roi in roi_sequence:
        roi_number = getattr(roi, "ROINumber", None)
        roi_name = str(getattr(roi, "ROIName", "Unnamed") or "Unnamed")
        obs = obs_by_number.get(int(roi_number)) if roi_number is not None else None
        structures.append({
            "roi_number": roi_number,
            "structure_name": roi_name,
            "interpreted_type": str(getattr(obs, "RTROIInterpretedType", "") or "") if obs is not None else "",
        })
    return structures
