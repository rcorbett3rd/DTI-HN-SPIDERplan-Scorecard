from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from dicom_parser import (
    classify_rt_files,
    extract_plan_summary,
    extract_structures,
    get_prescription_dose_gy,
    load_dicoms,
    save_uploaded_files,
)
from dvh_engine import calculate_dvh_metrics, dvh_note, global_hotspot_analysis
from scorecard_engine import build_metric_table, domain_scores, final_grade
from spider_chart import make_spider_chart


st.set_page_config(page_title="DTI - HN SPIDERplan Scorecard", layout="wide")


@st.cache_data
def load_config() -> dict:
    config_path = Path(__file__).parent / "scoring_config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _is_excluded_structure_name(name: str) -> bool:
    n = str(name).strip().lower()
    return n.startswith("z")


def _is_ln_helper_structure(name: str) -> bool:
    n = str(name).strip().lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", n) if t]
    return "ln" in tokens or n.startswith("ln") or n.endswith("ln")


def _is_target_name(name: str) -> bool:
    n = str(name).strip().lower()
    if _is_excluded_structure_name(name) or _is_ln_helper_structure(name):
        return False
    # Optimization helper target contours such as PTV56_Opti are not score targets.
    if n.endswith("opti"):
        return False
    return any(k in n for k in ["ptv", "ctv", "gtv"])


def _is_eval_structure(name: str) -> bool:
    """Only structures ending in _eval are evaluation structures."""
    return str(name).strip().lower().endswith("_eval")


def _parent_from_eval_name(name: str) -> str:
    s = str(name).strip()
    return re.sub(r"_eval$", "", s, flags=re.IGNORECASE)


def _name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _target_type(name: str) -> str:
    n = str(name).lower()
    suffix = "_eval" if _is_eval_structure(name) else ""
    if "ptv" in n:
        return f"PTV{suffix}"
    if "ctv" in n:
        return f"CTV{suffix}"
    if "gtv" in n:
        return f"GTV{suffix}"
    return f"Target{suffix}"


def _rx_values_from_plan(rtplan) -> list[float]:
    doses: list[float] = []
    for ref in getattr(rtplan, "DoseReferenceSequence", []) or []:
        dose = getattr(ref, "TargetPrescriptionDose", None)
        if dose is None:
            continue
        try:
            d = float(dose)
            if d > 0 and d not in doses:
                doses.append(d)
        except Exception:
            pass
    return sorted(doses, reverse=True)


def _infer_rx_from_name(name: str, known_rx: list[float]) -> tuple[float | None, str]:
    base = _parent_from_eval_name(name) if _is_eval_structure(name) else str(name)
    n_raw = base.lower()
    n = n_raw.replace("cgy", " cgy").replace("gy", " gy")

    m = re.search(r"(\d{3,5}(?:\.\d+)?)\s*cgy", n)
    if m:
        return float(m.group(1)) / 100.0, "name cGy"

    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*gy", n)
    if m:
        return float(m.group(1)), "name Gy"

    m = re.search(r"(?:ptv|ctv|gtv)[_\-\s]*(?:high|mid|low|boost)?[_\-\s]*(\d{2,5}(?:\.\d+)?)", n)
    if not m:
        m = re.search(r"(\d{2,5}(?:\.\d+)?)", n)
    if m:
        raw = float(m.group(1))
        value = raw / 100.0 if raw >= 1000 else raw
        if 10 <= value <= 120:
            return value, "name pattern"

    # In HN SIB plans, GTVs without a numeric label are treated as the highest/Rx dose level.
    if "gtv" in n_raw and known_rx:
        return max(known_rx), "GTV assigned highest plan Rx"

    if len(known_rx) == 1:
        return known_rx[0], "single plan Rx"
    return None, "manual required"


def build_target_rx_table(structure_df: pd.DataFrame, rtplan) -> pd.DataFrame:
    """
    target logic:
    - PTV/CTV/GTV structures receive their own assigned Rx.
    - Structures ending exactly in _eval inherit Rx from the same parent structure name without _eval.
    - _eval rows are scored only for V105% in scorecard_engine.py.
    """
    known_rx = _rx_values_from_plan(rtplan)
    cols = ["structure", "target_type", "parent_structure", "assigned_rx_gy", "rx_source", "scoring_role", "include_in_score"]
    if structure_df is None or structure_df.empty or "structure_name" not in structure_df.columns:
        return pd.DataFrame(columns=cols)

    target_names = [str(x) for x in structure_df["structure_name"].tolist() if _is_target_name(str(x))]
    all_name_keys = {_name_key(n): n for n in target_names}
    rows: list[dict] = []
    base_rx_by_key: dict[str, float] = {}

    # First pass: non-eval targets. These are the parent dose levels.
    for name in target_names:
        if _is_eval_structure(name):
            continue
        rx, source = _infer_rx_from_name(name, known_rx)
        if rx is not None:
            base_rx_by_key[_name_key(name)] = rx
        rows.append({
            "structure": name,
            "target_type": _target_type(name),
            "parent_structure": "",
            "assigned_rx_gy": rx,
            "rx_source": source,
            "scoring_role": "Coverage / target-dose quality",
            "include_in_score": True,
        })

    # Second pass: *_eval structures. They inherit from exact same name minus _eval.
    for name in target_names:
        if not _is_eval_structure(name):
            continue
        parent = _parent_from_eval_name(name)
        parent_key = _name_key(parent)
        parent_display = all_name_keys.get(parent_key, parent)
        rx = base_rx_by_key.get(parent_key)
        source = "inherited from parent target"
        if rx is None:
            rx, source = _infer_rx_from_name(parent, known_rx)
            if rx is not None:
                source = "inferred from parent name"
        rows.append({
            "structure": name,
            "target_type": _target_type(name),
            "parent_structure": parent_display,
            "assigned_rx_gy": rx,
            "rx_source": source if rx is not None else "manual required - parent Rx missing",
            "scoring_role": "V105% hotspot review only",
            "include_in_score": True,
        })

    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out["sort_key"] = out["structure"].str.lower().str.replace("_eval", "zz_eval", regex=False)
        out = out.sort_values(["sort_key", "scoring_role"]).drop(columns=["sort_key"]).reset_index(drop=True)
    return out


config = load_config()

st.title("DTI - HN SPIDERplan Scorecard")
st.caption("Head and neck RP + RS + RD scorecard with per-target Rx assignment, PTV_eval V105% review, and global hotspot location screening.")

with st.expander("Clinical / security disclaimer", expanded=False):
    st.write(
        "This prototype is for research, development, and local plan-review support only. "
        "It is not a replacement for clinical TPS DVH review, physician approval, physicist QA, chart rounds, "
        "or institutional policy. Use only de-identified or institutionally approved datasets. "
        "Validate all DVH and scorecard outputs against Eclipse/ARIA or your clinical TPS before any clinical use."
    )

uploaded_files = st.file_uploader(
    "Upload RT Plan/RP, RT Structure/RS, and RT Dose/RD files",
    type=["dcm", "dicom", "DCM"],
    accept_multiple_files=True,
    help="Full DVH scoring requires RP + RS + RD. Missing or unreadable files show warnings instead of crashing.",
)

with st.sidebar:
    st.header("Options")
    structure_limit = st.number_input(
        "Structure calculation limit",
        min_value=1,
        max_value=300,
        value=120,
        step=5,
        help="Lower this if a very large structure set is slow. The app prioritizes target/eval structures before applying this limit.",
    )
    score_only_assigned_targets = st.checkbox(
        "Require Rx for scored targets",
        value=True,
        help="Recommended. Targets and *_eval structures without an assigned Rx are paused until you enter the correct Gy value or uncheck Score.",
    )

if not uploaded_files:
    st.info("Upload RP + RS + RD files to generate the SPIDERplan scorecard.")
    st.stop()

try:
    uploaded_paths = save_uploaded_files(uploaded_files, upload_dir=Path("uploaded_dicoms"))
    dicoms = load_dicoms(uploaded_paths)

    if not dicoms:
        st.error("No readable DICOM files were found. Please confirm the files are valid DICOM RT exports.")
        st.stop()

    files = classify_rt_files(dicoms)
    rtplan = files.get("RTPLAN")
    rtstruct = files.get("RTSTRUCT")
    rtdose = files.get("RTDOSE")
    other_files = files.get("OTHER", [])

    st.subheader("Detected DICOM Files")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RT Plan / RP", "Yes" if rtplan is not None else "No")
    c2.metric("RT Structure / RS", "Yes" if rtstruct is not None else "No")
    c3.metric("RT Dose / RD", "Yes" if rtdose is not None else "No")
    c4.metric("Other files", len(other_files))

    missing = []
    if rtplan is None:
        missing.append("RT Plan / RP")
    if rtstruct is None:
        missing.append("RT Structure / RS")
    if rtdose is None:
        missing.append("RT Dose / RD")

    if missing:
        st.error("Full SPIDERplan scoring requires: " + ", ".join(missing) + ".")
        st.info("Use the Readiness App for RP/RS pre-checks without RD. This Full DVH app waits until all required files are present.")
        st.stop()

    plan_summary = extract_plan_summary(rtplan)
    rx_dose_gy = get_prescription_dose_gy(rtplan)
    structures = extract_structures(rtstruct)

    plan_summary_df = pd.DataFrame([plan_summary]) if plan_summary else pd.DataFrame()
    structure_df = pd.DataFrame(structures)

    st.markdown("---")
    st.header("Target Prescription Assignment")
    st.caption(
        "Assign each PTV/CTV/GTV its own Rx dose. A structure ending exactly with `_eval` inherits the Rx from the matching parent name without `_eval` "
        "and is scored only for V105% hotspot review. Example: `PTV_7000_eval` inherits from `PTV_7000`."
    )

    target_rx_df = build_target_rx_table(structure_df, rtplan)
    if target_rx_df.empty:
        st.error("No PTV/CTV/GTV structures were detected. Target scoring cannot be performed.")
        st.stop()

    edited_target_rx_df = st.data_editor(
        target_rx_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "structure": st.column_config.TextColumn("Structure", disabled=True),
            "target_type": st.column_config.TextColumn("Target type", disabled=True),
            "parent_structure": st.column_config.TextColumn("Parent target", disabled=True),
            "assigned_rx_gy": st.column_config.NumberColumn("Assigned Rx Gy", min_value=0.0, max_value=120.0, step=0.1, format="%.1f"),
            "rx_source": st.column_config.TextColumn("Rx source", disabled=True),
            "scoring_role": st.column_config.TextColumn("Scoring role", disabled=True),
            "include_in_score": st.column_config.CheckboxColumn("Score", default=True),
        },
        key="target_rx_editor",
    )

    active_rx_df = edited_target_rx_df[edited_target_rx_df["include_in_score"] == True].copy()
    active_rx_df["assigned_rx_gy"] = pd.to_numeric(active_rx_df["assigned_rx_gy"], errors="coerce")
    missing_rx = active_rx_df[active_rx_df["assigned_rx_gy"].isna() | (active_rx_df["assigned_rx_gy"] <= 0)]

    if not missing_rx.empty and score_only_assigned_targets:
        st.warning("Some scored targets/eval structures do not have an assigned Rx. Enter the correct Gy value or uncheck Score for those rows.")
        st.dataframe(missing_rx[["structure", "target_type", "parent_structure", "assigned_rx_gy", "rx_source", "scoring_role"]], use_container_width=True, hide_index=True)
        st.stop()

    target_rx_map = {
        str(row["structure"]): float(row["assigned_rx_gy"])
        for _, row in active_rx_df.iterrows()
        if pd.notna(row["assigned_rx_gy"]) and float(row["assigned_rx_gy"]) > 0
    }
    eval_names = [s for s in target_rx_map if _is_eval_structure(s)]
    # BODY/External is not included in ordinary DVH/structure scoring. It is used only by the separate global max hotspot review.
    priority_structures = list(target_rx_map.keys())

    st.success(f"{len(target_rx_map)} target/eval structures have assigned Rx values. {len(eval_names)} `_eval` structure(s) will be scored for V105% only.")

    progress = st.progress(15, text="Files loaded. Calculating DVH metrics and scorecard...")

    with st.spinner("Calculating prototype DVH metrics from RT Dose and RT Structure contours..."):
        dvh_df, dvh_warnings = calculate_dvh_metrics(
            rtstruct=rtstruct,
            rtdose=rtdose,
            rx_dose_gy=None,
            structure_limit=int(structure_limit),
            rx_map=target_rx_map,
            priority_structures=priority_structures,
        )

    progress.progress(70, text="DVH metrics calculated. Building SPIDERplan scorecard...")

    if dvh_df is None or dvh_df.empty:
        st.error("No DVH metrics could be calculated. The app did not crash, but the RD/RS geometry may not align or pixel data may be unreadable.")
        if dvh_warnings:
            with st.expander("DVH calculation warnings", expanded=True):
                for warning in dvh_warnings[:200]:
                    st.write(f"- {warning}")
                if len(dvh_warnings) > 200:
                    st.write(f"...and {len(dvh_warnings) - 200} additional warnings.")
        st.stop()

    metric_df = build_metric_table(dvh_df, rx_dose_gy=rx_dose_gy)

    global_hotspot_df, global_hotspot_warnings = global_hotspot_analysis(rtstruct, rtdose, rx_map=target_rx_map)
    if global_hotspot_warnings:
        dvh_warnings.extend(global_hotspot_warnings)
    if global_hotspot_df is not None and not global_hotspot_df.empty:
        metric_df = pd.concat([metric_df, global_hotspot_df], ignore_index=True)

    domain_df = domain_scores(metric_df)
    grade = final_grade(domain_df)
    fig = make_spider_chart(domain_df)

    progress.progress(100, text="SPIDERplan scorecard ready.")
    progress.empty()

    st.markdown("---")
    st.header("SPIDERplan Scorecard Snapshot")
    st.caption("Immediate plan-quality overview using per-target Rx values, helper-contour exclusions, optic-structure scoring, PTV_eval V105% review, and a separate global max hotspot location review.")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Final SPIDERplan Score", f"{grade.get('score', 0)}")
    s2.metric("Final SPIDERplan Grade", str(grade.get("grade", "N/A")))
    s3.metric("Scored Structures", 0 if metric_df is None else len(metric_df[pd.to_numeric(metric_df.get("score"), errors="coerce").notna()]))
    s4.metric("PTV_eval Reviews", len(eval_names))

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Spider chart could not be generated because no domain scores were available.")

    if domain_df is not None and not domain_df.empty:
        with st.expander("Show domain scores", expanded=True):
            st.dataframe(domain_df, use_container_width=True, hide_index=True)

    if metric_df is not None and not metric_df.empty:
        priority_cols = ["structure", "category", "scoring_role", "assigned_rx_gy", "score", "grade", "notes"]
        available_cols = [c for c in priority_cols if c in metric_df.columns]
        attention_df = metric_df.copy()
        attention_df["score_numeric"] = pd.to_numeric(attention_df.get("score"), errors="coerce")
        attention_df = attention_df.sort_values("score_numeric", ascending=True).drop(columns=["score_numeric"])
        with st.expander("Lowest-scoring items to review first", expanded=True):
            st.dataframe(attention_df[available_cols].head(15), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.header("Detailed Review")

    with st.expander("Target Rx Assignment Table", expanded=True):
        st.dataframe(edited_target_rx_df, use_container_width=True, hide_index=True)

    with st.expander("Plan Summary", expanded=False):
        if not plan_summary_df.empty:
            st.dataframe(plan_summary_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Plan summary could not be extracted.")
        if rx_dose_gy is None:
            st.warning("No TargetPrescriptionDose was found in the RT Plan. Target scoring uses the per-structure Rx table above.")
        else:
            st.success(f"Highest prescription detected in RT Plan: {rx_dose_gy:g} Gy. scoring uses the per-structure Rx table, not one global prescription.")

    with st.expander("Structure List", expanded=False):
        if not structure_df.empty:
            st.dataframe(structure_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No structures were extracted from the RT Structure Set.")

    with st.expander("DVH / Dose Metrics", expanded=False):
        st.caption(dvh_note())
        st.dataframe(dvh_df, use_container_width=True, hide_index=True)

    with st.expander("Full Metric Scorecard", expanded=False):
        if metric_df is None or metric_df.empty:
            st.warning("No metric scorecard could be generated from the DVH metrics.")
        else:
            st.dataframe(metric_df, use_container_width=True, hide_index=True)

    if dvh_warnings:
        with st.expander("DVH calculation warnings", expanded=False):
            for warning in dvh_warnings[:200]:
                st.write(f"- {warning}")
            if len(dvh_warnings) > 200:
                st.write(f"...and {len(dvh_warnings) - 200} additional warnings.")

    st.subheader("Export")
    e1, e2, e3 = st.columns(3)
    e1.download_button("Download DVH CSV", data=_csv_bytes(dvh_df), file_name="spiderplan_dvh_metrics.csv", mime="text/csv")
    if metric_df is not None and not metric_df.empty:
        e2.download_button("Download Scorecard CSV", data=_csv_bytes(metric_df), file_name="spiderplan_metric_scorecard.csv", mime="text/csv")
    if domain_df is not None and not domain_df.empty:
        e3.download_button("Download Domain Scores CSV", data=_csv_bytes(domain_df), file_name="spiderplan_domain_scores.csv", mime="text/csv")

except Exception as e:
    st.error("The app hit an unexpected error, but it was caught safely.")
    st.exception(e)
    st.stop()
