# Decisions

One line per non-obvious choice, newest last. `[default]` = taken without a reply from the author
(Checkpoint rule); revisit if the author disagrees.

## Phase 0 — 2026-10-27
- Spectrum PDF fetched from the URL on the landing page; `othp109.pdf` serves HTML, not the report.
- Spectrum's label-ceiling figure is an 8–10% error rate with a qualitative ("moderate") Fleiss' kappa; RQ4 compares against the error rate, not against a kappa value (none is reported).
- RQ3 is positioned as a complement to Spectrum's N2 (back-testing indices built from web prices), quantifying its R4 remark on error impact by hierarchy level; it does not back-test web prices.
- Official public CPI data come from the TPSO Open API (documented endpoints only). The internal `/api/*` routes behind index.tpso.go.th's UI (CSRF-protected) are not used.
- Base-2566 weights are published only for 29 aggregates (division, food group, three transport lines, core/fresh/energy) as price-updated shares in the monthly release; base weights are backed out as w0 = s_t · I_total,t / I_i,t. Two months agree to within ±0.01 pp, and the results match the rebasing note's division weights.
- Base-2566 weights apply from Jan 2568; earlier months are a linked back-series built with base-2562 weights. Any check of the Laspeyres identity, or any RQ3 simulation that uses 2566 weights, uses Jan 2568 onwards (21 months to Sep 2569) or treats earlier months as linked.
- Weights cannot be recovered by regressing parent on child indices: with 21–45 months the system is ill-conditioned (NNLS reproduces the headline index to 0.004 points but misallocates small divisions by up to 1 pp). Implied weights can therefore be at most a labelled sensitivity, never a substitute for official weights.
- The official weights reproduce the published headline from the division indices to within 0.045 index points (2-decimal rounding). The "perfect classifier reproduces official inflation" test is therefore exact on our own aggregation and within rounding tolerance against the published series.
- Item-level (464 items) indices and weights are not public; the finest public index level is ~123 leaf codes (current and back-series codes mixed), of which 28 are in food. The classification target becomes the leaf index codes within the supermarket scope, with an item-level crosswalk kept for reporting.
