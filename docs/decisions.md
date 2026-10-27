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

## Checkpoint A replies — 2026-10-27
- LLM access goes through the local `claude` CLI (`claude -p --output-format json --tools "" --no-session-persistence`), not API keys (author's choice). Frontier = `claude-opus-5-5`, low-cost = `claude-haiku-5-5`. Each call's `total_cost_usd`, `usage` and `modelUsage` go to `results/llm_costs.csv`, and that reported cost counts against the USD 60 cap. Probe: ~4.4k fixed prompt tokens per call; Haiku ≈ USD 0.0009, Opus ≈ USD 0.036 per single-item call.
- Opus runs only on the gold set and a fixed subsample of the MOC evaluation set; Haiku and the cascade run on full evaluation sets (keeps spend well under the cap).
- Gold labels: 600 products, 150 of them labelled twice blind (author accepted the recommendation). The MOC official-spec set is the second evaluation set.
- Weights: official 29-aggregate base weights; within-aggregate leaf weights are an explicitly labelled sensitivity (equal split and constrained NNLS). If the author drops subgroup weights into `data/raw/tpso/manual/`, they replace the sensitivity.
- Paper toolchain: LaTeX via `tectonic` in `~/.local/bin` (no LaTeX on this machine), as MISSION specifies.
- (Superseded 2026-10-27, see Phase 1) LLM access via the `claude` CLI.

## Phase 1 — 2026-10-27
- LLM access switches to the Anthropic API with a key the author provided (stored in gitignored `.env` as `ANTHROPIC_API_KEY` and `ANTHROPIC_WORKSPACE_ID`, file mode 600; the key is not workspace-scoped, so every request sends `anthropic-workspace-id`). This avoids the CLI's ~4.4k fixed tokens per call. The key's balance (USD 200, per the author) does not change the MISSION cap: total spend stays ≤ USD 60 unless the author explicitly raises it. Probe: one `claude-haiku-5-5` call, 12 input / 4 output tokens.
- Pre-commit: detect-secrets with a baseline, plus a pygrep hook that blocks `sk-ant-…` keys and `wrkspc_…` ids, and a hook that blocks any `.env*` file.
- The backed-out w0 are the 2566 expenditure weights (they match the rebasing note: food 38.78 vs 38.74, fresh food 17.43 vs 17.43, other divisions within 0.04), but they do not add up the hierarchy (food − Σ groups = 0.11 pp). The index is a Lowe index with price reference Dec 2567, the link month. The shares that add up are the price-updated `w_pr = w0 · I_i,Dec2567 / I_total,Dec2567`: the hierarchy closes to ≤ 0.003 pp at Dec 2567, compared with 0.04–0.11 pp for other candidate reference periods. All aggregation, cells and the RQ3 simulator use `w_pr` with relatives I_t / I_Dec2567. `w0` is kept for reporting.
- Weight cells: each leaf belongs to its deepest official ancestor-or-self. Two cells are residuals: 11400 minus (11411, 11421) = 0.457 pp (11412 dried/processed vegetables, 11422 processed fruit), and 50000 minus (51000, 52200, 54000) = 8.97 pp (6 leaves). Cell weights total 99.995 (rounding of the printed shares).
- Within-cell leaf weights come in two labelled sensitivities: equal split, and sum-constrained NNLS on relatives Jan 2568–Sep 2569. NNLS reproduces the published headline relative to within 0.012% and equal split to within 0.28%; NNLS weights inside a cell are still ill-identified (for example flour 11121 gets 0.02 pp).
