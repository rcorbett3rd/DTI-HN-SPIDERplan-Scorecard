# DTI - HN SPIDERplan Scorecard

Head and neck SPIDERplan scorecard prototype for RP + RS + RD DICOM RT exports.

## Current scoring behavior

- Per-target Rx assignment for PTV/CTV/GTV structures.
- Target optimization helpers ending in `opti` are not scored.
- Structures starting with `z` and LN helper contours are not scored.
- `optic` structures are scored as OARs.
- BODY/External is not scored as a structure. It is only used internally for the separate global max hotspot review.
- Target scoring now includes `V100Rx_%`, `V95Rx_%`, and `D0.03cc_%Rx`.
- The highest-dose PTV does not require an `_eval` structure for V105%; V105% is evaluated directly on that PTV.
- Lower-dose `_eval` structures inherit Rx from the parent target and are scored only for V105%.
- OAR scoring uses scalable preferred/acceptable ranges instead of hard 100/85/70/30 buckets.

## Scoring scale

For upper-limit constraints where lower is better:

- Non-variable constraints: ideal or better = 100; ideal to preferred = 100 to 90; above preferred = 0.
- Variable acceptable constraints: below preferred = 100; at preferred = 90; preferred to acceptable = 90 to 50; above acceptable = 0.

For target coverage:

- `V100Rx_% >= 95%` = 100.
- If `V100Rx_% < 95%` but `V95Rx_% >= 95%`, the score scales from 90 to 99.9 based on V100Rx.
- If `V95Rx_% < 95%`, score = 0.

For PTV/eval V105%:

- `<=5%` = 100.
- `5-10%` scales from 100 to 90.
- `10-20%` scales from 90 to 50.
- `>=20%` = 0.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer

Prototype only. Validate all DVH and scorecard outputs against the clinical TPS before any clinical or patient-care use.


## Latest scoring correction

Target coverage is scored against each structure's own assigned prescription dose. Lower-dose SIB target D0.03cc and V105% are no longer used to fail the coverage row, because those targets may intentionally overlap higher-dose target volumes. Lower-dose hotspot review remains assigned to matching `_eval` structures, while the highest-dose PTV is evaluated directly for V105% and D0.03cc. Dmin is reported as a screen but does not hard-fail a target when V95Rx remains acceptable.
