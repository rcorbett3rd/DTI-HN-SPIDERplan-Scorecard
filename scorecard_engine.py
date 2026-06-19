from __future__ import annotations

from typing import Any
import re
import math
import pandas as pd
import numpy as np


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(name).strip().lower()) if t]


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _is_ln_helper_structure(name: str) -> bool:
    n = str(name).strip().lower()
    toks = _tokens(name)
    return "ln" in toks or n.startswith("ln") or n.endswith("ln")


def _is_body_or_external(name: str) -> bool:
    n = str(name).strip().lower()
    return n in {"body", "external"} or n.startswith("external")


def _is_excluded_helper_structure(name: str) -> bool:
    # Optic/Opti distinction matters: optic OARs are scored; target optimization helpers ending in opti are not.
    n = str(name).strip().lower()
    return n.startswith("z") or _is_ln_helper_structure(name) or _is_body_or_external(name)


def _is_eval_structure(name: str) -> bool:
    return str(name).strip().lower().endswith("_eval")


def _is_target(name: str) -> bool:
    n = str(name).strip().lower()
    if _is_excluded_helper_structure(name):
        return False
    if n.endswith("opti"):
        return False
    return any(k in n for k in ["ptv", "ctv", "gtv"])


def _is_ptv(name: str) -> bool:
    return "ptv" in str(name).strip().lower() and _is_target(name)


def _is_serial_oar(name: str) -> bool:
    n = str(name).lower()
    return any(k in n for k in ["cord", "brainstem", "chiasm", "optic", "retina", "lens"])


def _grade(score: float) -> str:
    return "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"


def _finite_float(x: Any) -> float | None:
    try:
        if pd.isna(x):
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _lerp(value: float, x0: float, y0: float, x1: float, y1: float) -> float:
    if x1 == x0:
        return min(y0, y1)
    t = (value - x0) / (x1 - x0)
    t = max(0.0, min(1.0, t))
    return y0 + t * (y1 - y0)


def score_upper_limit(value: Any, preferred: float, acceptable: float | None = None, ideal: float | None = None) -> float:
    """Score constraints where lower is better: mean dose, D0.03cc, D5%, V30Gy, V105%.

    Non-variable acceptance:
        ideal or better = 100; ideal -> preferred scales 100 -> 90; > preferred = 0.
    Variable acceptance:
        below preferred = 100; at preferred = 90; preferred -> acceptable scales 90 -> 50; > acceptable = 0.
    """
    v = _finite_float(value)
    if v is None:
        return 0.0
    p = float(preferred)
    a = None if acceptable is None else float(acceptable)

    if a is None or a <= p:
        i = float(ideal) if ideal is not None else 0.80 * p
        if v <= i:
            return 100.0
        if v <= p:
            return round(_lerp(v, i, 100.0, p, 90.0), 1)
        return 0.0

    if v < p:
        return 100.0
    if v <= a:
        return round(_lerp(v, p, 90.0, a, 50.0), 1)
    return 0.0


def score_v105(value: Any) -> float:
    """PTV/eval hotspot score: <=5 = 100, 5-10 = 100->90, 10-20 = 90->50, >=20 = 0."""
    v = _finite_float(value)
    if v is None:
        return 0.0
    if v <= 5:
        return 100.0
    if v < 10:
        return round(_lerp(v, 5.0, 100.0, 10.0, 90.0), 1)
    if v < 20:
        return round(_lerp(v, 10.0, 90.0, 20.0, 50.0), 1)
    return 0.0


def score_target_coverage(v100: Any, v95: Any) -> float:
    """Target coverage scoring against that structure's own assigned Rx dose.

    Desired target coverage ladder:
      1) V100Rx >= 100% of target volume -> 100
      2) V100Rx 95% to <100% -> linearly scales 90 to 99
      3) V100Rx <95% but V95Rx >=95% -> linearly scales 70 to 90 by V100Rx
      4) V95Rx <95% -> 0

    In plain language: full prescription-dose coverage earns 100; coverage that
    remains clinically acceptable but less ideal is graded by how much of the
    target receives 100% Rx, with V95Rx>=95% as the minimum acceptable floor.
    """
    v100f = _finite_float(v100)
    v95f = _finite_float(v95)

    if v100f is None or v95f is None:
        return 0.0
    if v100f >= 100.0:
        return 100.0
    if v100f >= 95.0:
        return round(_lerp(v100f, 95.0, 90.0, 100.0, 99.0), 1)
    if v95f >= 95.0:
        # Between ideal V100 coverage and minimum acceptable V95 coverage.
        # At V100=95, score is 90. As V100 falls away from 95 but V95 remains
        # acceptable, score scales down toward 70.
        bounded_v100 = max(0.0, min(95.0, v100f))
        return round(_lerp(bounded_v100, 0.0, 70.0, 95.0, 90.0), 1)
    return 0.0


def score_target_min_dose(dmin_pct_rx: Any) -> float:
    """Target minimum dose score using each PTV/CTV/GTV's assigned Rx.

    >93% Rx = 100
    85% to 93% Rx = 90 to 95 scaled upward
    <85% Rx = 0
    """
    v = _finite_float(dmin_pct_rx)
    if v is None:
        return 0.0
    if v > 93.0:
        return 100.0
    if v >= 85.0:
        return round(_lerp(v, 85.0, 90.0, 93.0, 95.0), 1)
    return 0.0


def _score_eval_target(row: pd.Series) -> tuple[float, str]:
    assigned_rx = _finite_float(row.get("assigned_rx_gy"))
    v105 = row.get("V105Rx_%")
    if assigned_rx is None or assigned_rx <= 0:
        return 0.0, "PTV_eval not scored: no assigned/inherited Rx dose"
    if _finite_float(v105) is None:
        return 0.0, "PTV_eval V105% unavailable; review manually"
    score = score_v105(v105)
    return score, f"PTV_eval V105%={float(v105):.1f}% scored on 5/10/20% scale"


def _score_target_coverage(row: pd.Series, highest_rx: float | None = None) -> tuple[float, str]:
    assigned_rx = _finite_float(row.get("assigned_rx_gy"))
    if assigned_rx is None or assigned_rx <= 0:
        return 0.0, "Target not scored: no assigned Rx dose"

    name = str(row.get("structure", ""))
    is_highest_ptv = bool(
        highest_rx is not None
        and abs(float(assigned_rx) - float(highest_rx)) < 0.05
        and _is_ptv(name)
    )

    v100 = row.get("V100Rx_%")
    v95 = row.get("V95Rx_%")
    d003_pct = row.get("D0.03cc_%Rx")
    dmin_pct = row.get("Dmin_%Rx")

    # Coverage is the primary target score and is ALWAYS calculated against
    # this target's own assigned Rx. This prevents 56/63 Gy targets from being
    # incorrectly judged against the 70 Gy dose level.
    cov_score = score_target_coverage(v100, v95)

    notes = [
        f"Coverage by assigned Rx {assigned_rx:g} Gy: V100Rx={_finite_float(v100)}%, V95Rx={_finite_float(v95)}%; coverage score={cov_score:.1f}"
    ]

    # Dmin can be unstable from contour edge sampling. It is reported and used
    # only as a warning modifier, not as an automatic row-failing hard stop when
    # V95Rx remains acceptable.
    min_score = score_target_min_dose(dmin_pct)
    dmin_f = _finite_float(dmin_pct)
    notes.append(f"Dmin={dmin_f}%Rx; min-dose screen={min_score:.1f} (reported, not hard-fail if V95Rx passes)")

    scores = [cov_score]

    # High-dose quality metrics are only scored on the highest-dose PTV itself.
    # Lower-dose PTVs/CTVs often overlap higher-dose regions in SIB plans, so
    # their D0.03cc/V105% should not fail coverage. Lower-dose hotspot review is
    # handled by matching *_eval structures.
    if is_highest_ptv:
        if _finite_float(d003_pct) is not None:
            d003_score = score_upper_limit(d003_pct, preferred=110.0, acceptable=115.0)
            scores.append(d003_score)
            notes.append(f"Highest-dose PTV D0.03cc={float(d003_pct):.1f}%Rx scored with 110/115% scale")
        v105 = row.get("V105Rx_%")
        if _finite_float(v105) is not None:
            h_score = score_v105(v105)
            scores.append(h_score)
            notes.append(f"Highest-dose PTV V105Rx={float(v105):.1f}% scored on PTV itself")

    # If coverage fails the V95 floor, the target fails. Otherwise use the most
    # conservative scored target component without allowing lower-dose SIB
    # overlap metrics to create false failures.
    score = min(scores) if scores else 0.0
    return round(score, 1), "; ".join(notes)


# CoH HN template preferred limits, with review/acceptable thresholds where the template comment provides them
# or where prior institutional/Timmerman-style serial-organ review commonly benefits from a narrow variable band.
OAR_RULES: list[dict[str, Any]] = [
    {"keys": ["spinalcord_prv", "cord_prv"], "metric": "D0.03cc_Gy", "preferred": 50, "acceptable": None, "ideal": 45, "label": "SpinalCord_PRV D0.03cc"},
    {"keys": ["spinalcord", "cord"], "metric": "D0.03cc_Gy", "preferred": 45, "acceptable": None, "ideal": 40, "label": "SpinalCord D0.03cc"},
    {"keys": ["brainstem_prv"], "metric": "D0.03cc_Gy", "preferred": 58, "acceptable": 60, "label": "Brainstem_PRV D0.03cc"},
    {"keys": ["brainstem"], "metric": "D0.03cc_Gy", "preferred": 54, "acceptable": 58, "label": "Brainstem D0.03cc"},
    {"keys": ["opticnrv_prv", "opticchiasm_prv"], "metric": "D0.03cc_Gy", "preferred": 54, "acceptable": None, "ideal": 50, "label": "Optic PRV D0.03cc"},
    {"keys": ["opticnrv", "optic_chiasm", "opticchiasm", "chiasm"], "metric": "D0.03cc_Gy", "preferred": 50, "acceptable": 54, "label": "Optic/chiasm D0.03cc"},
    {"keys": ["retina"], "metric": "D0.03cc_Gy", "preferred": 45, "acceptable": None, "ideal": 40, "label": "Retina D0.03cc"},
    {"keys": ["lens"], "metric": "D0.03cc_Gy", "preferred": 10, "acceptable": None, "ideal": 5, "label": "Lens D0.03cc"},
    {"keys": ["brain"], "metric": "D0.03cc_Gy", "preferred": 60, "acceptable": None, "ideal": 54, "label": "Brain D0.03cc"},
    {"keys": ["brachialplex"], "metric": "D0.03cc_Gy", "preferred": 66, "acceptable": None, "ideal": 60, "label": "Brachial plexus D0.03cc"},
    {"keys": ["parotid"], "metric": "Dmean_Gy", "preferred": 26, "acceptable": None, "ideal": 20, "label": "Parotid mean"},
    {"keys": ["parotid"], "metric": "V30Gy_%", "preferred": 50, "acceptable": None, "ideal": 40, "label": "Parotid V30Gy"},
    {"keys": ["glnd_submand", "submand"], "metric": "Dmean_Gy", "preferred": 39, "acceptable": None, "ideal": 30, "label": "Submandibular mean"},
    {"keys": ["cochlea"], "metric": "Dmean_Gy", "preferred": 35, "acceptable": 45, "label": "Cochlea mean"},
    {"keys": ["cochlea"], "metric": "D5_Gy", "preferred": 55, "acceptable": None, "ideal": 50, "label": "Cochlea D5%"},
    {"keys": ["musc_constrict_s", "constrict_s"], "metric": "Dmean_Gy", "preferred": 55, "acceptable": None, "ideal": 45, "label": "Superior constrictor mean"},
    {"keys": ["musc_constrict_m", "constrict_m", "musc_constrict_i", "constrict_i"], "metric": "Dmean_Gy", "preferred": 45, "acceptable": None, "ideal": 35, "label": "Middle/inferior constrictor mean"},
    {"keys": ["esophagus"], "metric": "Dmean_Gy", "preferred": 35, "acceptable": None, "ideal": 28, "label": "Esophagus mean"},
    {"keys": ["lips"], "metric": "Dmean_Gy", "preferred": 20, "acceptable": None, "ideal": 15, "label": "Lips mean"},
    {"keys": ["cavity_oral", "oralcavity", "oral_cavity"], "metric": "Dmean_Gy", "preferred": 35, "acceptable": None, "ideal": 28, "label": "Oral cavity mean"},
    {"keys": ["bone_mandible", "mandible"], "metric": "D0.03cc_Gy", "preferred": 71, "acceptable": None, "ideal": 66, "label": "Mandible D0.03cc"},
    {"keys": ["eyes", "eye_", "eye"], "metric": "D0.03cc_Gy", "preferred": 45, "acceptable": None, "ideal": 40, "label": "Eyes D0.03cc"},
    {"keys": ["mouth_floor", "floorofmouth"], "metric": "Dmean_Gy", "preferred": 40, "acceptable": None, "ideal": 32, "label": "Mouth floor mean"},
    {"keys": ["larynx"], "metric": "Dmean_Gy", "preferred": 35, "acceptable": None, "ideal": 28, "label": "Larynx mean"},
    {"keys": ["lobe_temporal", "temporal"], "metric": "D0.03cc_Gy", "preferred": 70, "acceptable": None, "ideal": 65, "label": "Temporal lobe D0.03cc"},
]


def _matched_oar_rules(name: str) -> list[dict[str, Any]]:
    n = _norm_name(name)
    matched = []
    for rule in OAR_RULES:
        if any(k in n for k in rule["keys"]):
            matched.append(rule)
    return matched


def _score_oar(row: pd.Series) -> tuple[float, str]:
    name = str(row.get("structure", ""))
    rules = _matched_oar_rules(name)
    if not rules:
        return None, "No configured HN OAR constraint matched; not included in score"

    scores = []
    notes = []
    for rule in rules:
        metric = rule["metric"]
        value = row.get(metric)
        preferred = float(rule["preferred"])
        acceptable = rule.get("acceptable")
        ideal = rule.get("ideal")
        label = rule["label"]
        suffix = "%" if metric.endswith("_%") else "Gy"
        v = _finite_float(value)
        if v is None:
            scores.append(0.0)
            notes.append(f"{label} unavailable; unable to score")
            continue
        s = score_upper_limit(v, preferred=preferred, acceptable=acceptable, ideal=ideal)
        scores.append(s)
        if acceptable is None or float(acceptable) <= preferred:
            notes.append(f"{label}: {v:.1f}{suffix}; ideal≤{ideal if ideal is not None else 0.8*preferred:g}, preferred≤{preferred:g}; score {s:.1f}")
        else:
            notes.append(f"{label}: {v:.1f}{suffix}; preferred<{preferred:g}, acceptable≤{float(acceptable):g}; score {s:.1f}")
    return min(scores), "; ".join(notes)


def _highest_non_eval_ptv_rx(dvh_df: pd.DataFrame) -> float | None:
    vals = []
    for _, r in dvh_df.iterrows():
        name = str(r.get("structure", ""))
        rx = _finite_float(r.get("assigned_rx_gy"))
        if rx is not None and rx > 0 and _is_ptv(name) and not _is_eval_structure(name):
            vals.append(rx)
    return max(vals) if vals else None


def _eval_parent_rx(name: str) -> str:
    return re.sub(r"_eval$", "", str(name).strip(), flags=re.IGNORECASE)


def build_metric_table(dvh_df: pd.DataFrame, rx_dose_gy: float | None) -> pd.DataFrame:
    if dvh_df is None or dvh_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    highest_ptv_rx = _highest_non_eval_ptv_rx(dvh_df)

    for _, row in dvh_df.iterrows():
        name = str(row.get("structure", ""))
        status = row.get("status", "")

        if _is_body_or_external(name):
            continue

        if status != "Calculated":
            rows.append({"structure": name, "category": "Not scored", "scoring_role": "Unavailable", "score": None, "grade": "N/A", "notes": status})
            continue

        if _is_excluded_helper_structure(name):
            rows.append({
                "structure": name,
                "category": "Not scored",
                "scoring_role": "Helper/planning contour ignored",
                "assigned_rx_gy": row.get("assigned_rx_gy"),
                "score": None,
                "grade": "N/A",
                "notes": "Excluded: structure starts with z, is BODY/External, or is an LN helper contour",
            })
            continue

        # Highest-dose PTV uses the PTV itself for V105. Matching highest-dose *_eval rows are ignored to avoid duplicate hotspot scoring.
        if _is_eval_structure(name):
            arx = _finite_float(row.get("assigned_rx_gy"))
            if highest_ptv_rx is not None and arx is not None and abs(arx - highest_ptv_rx) < 0.05:
                rows.append({
                    "structure": name,
                    "category": "Not scored",
                    "scoring_role": "Duplicate highest-dose eval ignored",
                    "assigned_rx_gy": arx,
                    "score": None,
                    "grade": "N/A",
                    "notes": f"Highest-dose PTV V105% is evaluated on parent PTV, not {_eval_parent_rx(name)}_eval",
                })
                continue

        if _is_target(name):
            assigned_rx = _finite_float(row.get("assigned_rx_gy"))
            if assigned_rx is None or assigned_rx <= 0:
                rows.append({
                    "structure": name,
                    "category": "Not scored",
                    "scoring_role": "Missing Rx",
                    "assigned_rx_gy": row.get("assigned_rx_gy"),
                    "score": None,
                    "grade": "N/A",
                    "notes": "Target not scored: no assigned Rx dose",
                })
                continue
            if _is_eval_structure(name):
                score, notes = _score_eval_target(row)
                category = "PTV_eval Hotspot Review"
                scoring_role = "V105% only"
            else:
                score, notes = _score_target_coverage(row, highest_ptv_rx)
                category = "Target Coverage / Dose Quality"
                scoring_role = "Coverage / target-dose quality"
        elif _is_serial_oar(name):
            score, notes = _score_oar(row)
            if score is None:
                rows.append({
                    "structure": name,
                    "category": "Not scored",
                    "scoring_role": "No configured constraint",
                    "assigned_rx_gy": row.get("assigned_rx_gy"),
                    "score": None,
                    "grade": "N/A",
                    "notes": notes,
                })
                continue
            category = "Serial Neurologic OAR"
            scoring_role = "OAR constraint screen"
        else:
            score, notes = _score_oar(row)
            if score is None:
                rows.append({
                    "structure": name,
                    "category": "Not scored",
                    "scoring_role": "No configured constraint",
                    "assigned_rx_gy": row.get("assigned_rx_gy"),
                    "score": None,
                    "grade": "N/A",
                    "notes": notes,
                })
                continue
            category = "General OAR / Other"
            scoring_role = "OAR constraint screen"

        rows.append({
            "structure": name,
            "category": category,
            "scoring_role": scoring_role,
            "assigned_rx_gy": row.get("assigned_rx_gy"),
            "D95_%Rx": row.get("D95_%Rx"),
            "Dmin_%Rx": row.get("Dmin_%Rx"),
            "V100Rx_%": row.get("V100Rx_%"),
            "V95Rx_%": row.get("V95Rx_%"),
            "V105Rx_%": row.get("V105Rx_%"),
            "D0.03cc_Gy": row.get("D0.03cc_Gy"),
            "D0.03cc_%Rx": row.get("D0.03cc_%Rx"),
            "Dmean_Gy": row.get("Dmean_Gy"),
            "V30Gy_%": row.get("V30Gy_%"),
            "D5_Gy": row.get("D5_Gy"),
            "score": round(float(score), 1),
            "grade": _grade(float(score)),
            "notes": notes,
        })
    return pd.DataFrame(rows)


def domain_scores(metric_df: pd.DataFrame) -> pd.DataFrame:
    if metric_df is None or metric_df.empty:
        return pd.DataFrame()
    scored = metric_df[pd.to_numeric(metric_df["score"], errors="coerce").notna()].copy()
    scored = scored[scored["category"] != "Not scored"]
    if scored.empty:
        return pd.DataFrame()
    grouped = scored.groupby("category", dropna=False)["score"].mean().reset_index()
    grouped = grouped.rename(columns={"category": "domain", "score": "domain_score"})
    grouped["domain_score"] = grouped["domain_score"].round(1)
    return grouped


def final_grade(domain_df: pd.DataFrame) -> dict[str, Any]:
    if domain_df is None or domain_df.empty:
        return {"score": 0.0, "grade": "N/A"}
    weights = {
        "Target Coverage / Dose Quality": 0.40,
        "PTV_eval Hotspot Review": 0.20,
        "Global Max Hotspot Review": 0.15,
        "Serial Neurologic OAR": 0.25,
        "General OAR / Other": 0.15,
    }
    total_weight = 0.0
    weighted = 0.0
    for _, row in domain_df.iterrows():
        domain = str(row["domain"])
        score = float(row["domain_score"])
        w = weights.get(domain, 0.10)
        weighted += score * w
        total_weight += w
    score = weighted / total_weight if total_weight else 0.0
    return {"score": round(score, 1), "grade": _grade(score)}
