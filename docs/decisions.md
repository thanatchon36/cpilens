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
- LLM access switches to the Anthropic API with a key the author provided (stored in gitignored `.env` as `ANTHROPIC_API_KEY` and `ANTHROPIC_WORKSPACE_ID`, file mode 600; the key is not workspace-scoped, so every request sends `anthropic-workspace-id`). This avoids the CLI's ~4.4k fixed tokens per call. The key's balance does not change the MISSION cap: total spend stays ≤ USD 60 unless the author explicitly raises it. Probe: one `claude-haiku-5-5` call, 12 input / 4 output tokens.
- Pre-commit: detect-secrets with a baseline, plus a pygrep hook that blocks `sk-ant-…` keys and `wrkspc_…` ids, and a hook that blocks any `.env*` file.
- The backed-out w0 are the 2566 expenditure weights (they match the rebasing note: food 38.78 vs 38.74, fresh food 17.43 vs 17.43, other divisions within 0.04), but they do not add up the hierarchy (food − Σ groups = 0.11 pp). The index is a Lowe index with price reference Dec 2567, the link month. The shares that add up are the price-updated `w_pr = w0 · I_i,Dec2567 / I_total,Dec2567`: the hierarchy closes to ≤ 0.003 pp at Dec 2567, compared with 0.04–0.11 pp for other candidate reference periods. All aggregation, cells and the RQ3 simulator use `w_pr` with relatives I_t / I_Dec2567. `w0` is kept for reporting.
- Weight cells: each leaf belongs to its deepest official ancestor-or-self. Two cells are residuals: 11400 minus (11411, 11421) = 0.457 pp (11412 dried/processed vegetables, 11422 processed fruit), and 50000 minus (51000, 52200, 54000) = 8.97 pp (6 leaves). Cell weights total 99.995 (rounding of the printed shares).
- Within-cell leaf weights come in two labelled sensitivities: equal split, and sum-constrained NNLS on relatives Jan 2568–Sep 2569. NNLS reproduces the published headline relative to within 0.012% and equal split to within 0.28%; NNLS weights inside a cell are still ill-identified (for example flour 11121 gets 0.02 pp).
- Scope: 28 labels plus OOS. These are the 21 food-at-home leaves, 12300 ready-to-eat (kept in scope and reported separately, because packaged ready meals are ambiguous between 12300 and food-at-home), 42100 personal care, 36000 cleaning, 71000 tobacco, and 72100/72200/72300 beer/wine/spirits. COICOP 2018 classes were checked against the UNSD class list (pp 43–52); tobacco is 02.3.0 in COICOP 2018, not 02.2.
- The in-scope weight depends on the within-cell method: 40.5 pp (equal split) vs 43.5 pp (NNLS), mostly because 42100 and 36000 sit in the large 40000 and 30000 cells. RQ3 reports both.
- The OFF→CPI crosswalk follows TPSO's own item headings (from the MOC specs), not COICOP intuition. For example, snacks and tofu are 11122, soy and other plant milks are 11320, dietary supplements are 11600, and dried chilli, garlic and peanuts are 11412. Packaged fish defaults to processed (11234), because fresh fish is rare in OFF.
- Silver-label rule: expand tags with taxonomy ancestors, then the highest `priority` wins, then the deepest tag; ties or `?` give no label. Priority 1 marks product-type groups (snacks, bakery, sweets, beverages, condiments, ready meals), because depth across OFF branches is not comparable (for example, potato crisps sit deeper under vegetables than under snacks).
- MOC specs: model input is the specification text only. The TPSO item heading is kept as metadata and never used as input, because it leaks the label. Specs under codes that are not current leaves (1,203) are dropped. Leaves 11710 and 12300–12500 have no specs.
- OFF extract: Thailand (all products with a name), 12 Asian markets (name and category required) and a fixed 40k Europe sample (hash-ordered with seed 20261027) as the high-resource transfer source. The dumps are pinned to HF revision `3ac7e8b6`, whose file hashes match our downloads.
- Gold sample: frame = Thai-market products with a Thai-script name (7,303). Strata = the CPI weight cell of the silver label, plus `none`. Allocation = a floor of 10, then half by stratum size and half by CPI cell weight. Inclusion probabilities are stored for design-weighted estimates. 150 of the 600 are double-labelled, drawn at random. The silver label is never shown to the labeller.

## Phase 2 — 2026-10-27
- The labelling app is a stdlib HTTP server (127.0.0.1 only, per-session token on writes) plus one static page, not Streamlit/Gradio. The keyboard-first flow (type-ahead, Enter, Shift+Enter, PgUp/PgDn) needs client-side state that Streamlit's rerun model makes slow, and it adds no dependency. Throughput was checked in headless Chromium.
- What the labeller sees: Thai and English name, brand, quantity, and the OFF front image (590/600 have one). Stratum, silver label, source and inclusion probability are never sent to the page. Search aliases are TPSO's own item headings from the MOC specs plus the `includes` text in `taxonomy/thcpi_scope.csv`. These are taxonomy definitions, not predictions.
- Labels are 28 leaves, `OOS` (out of scope) and `UNK` (cannot tell from the information shown), plus an `unsure` flag on any label. `UNK` items are excluded from accuracy metrics and counted separately. `unsure` is reported in RQ4.
- Double labelling (MISSION §3.3): pass 2 covers the 150 `double` products. It runs in a different seeded order (seed+2), never shows pass-1 labels, and offers a product only once its pass-1 label is ≥ 48 h old. The second annotator is the author, at least 48 h later. That gives intra-annotator agreement, which is a lower bound on inter-annotator disagreement, and the paper says so.
- Files are append-only: `data/gold/labels_pass{1,2}.csv`. The latest row per product is the label, and earlier rows are the audit trail.

## Phase 3 — 2026-10-27 (models; gold-independent parts)
- Text: one normalisation for every model (NFC, HTML entities, invisible characters, spacing). Model input is `name | brand | quantity`. Brands are de-duplicated case-insensitively. The brand/quantity ablation switches the fields off.
- Embeddings: `BAAI/bge-m3` and `intfloat/multilingual-e5-large` (with its `query: ` prefix), run locally on one RTX 4070 Ti. They are L2-normalised float32 and cached under `.cache/emb/` by model and SHA-1 of the text. Encoding the 88,622-text corpus took 197 s (bge-m3) and 183 s (e5-large). No API embedding model: the budget goes to LLM classification instead.
- Corpus and splits: OFF rows split 90/10 by a salted hash of the id. MOC specs split 60/20/20 by 7-digit item code, so specs of one item never straddle train and test. Every gold barcode, and every training text equal to a gold text, is removed from the training pool.
- RQ1 scenarios: `nonthai` (zero-shot cross-lingual), `+thaiweak`, `+moc{n}`, and `moc{n}` alone. The official-label budget `n` counts every MOC label a run touches: hash-ordered, then split 80/20 into train and validation. A small-n run therefore cannot tune on thousands of extra official labels.
- Classifiers: cosine k-NN (similarity-weighted votes; k ∈ {1,3,5,10,20,50} by validation macro recall), multinomial logistic regression, and a one-hidden-layer MLP (512, GELU, dropout 0.2). Linear and MLP share one trainer: Adam 1e-3, weight decay 1e-4, class weights ∝ 1/√frequency, early stopping on validation loss. Validation is checked every max(one pass, 50 steps) with batch ≤ n/4. A first grid evaluated once per epoch, which gave the 100–1,000-label runs only a handful of steps and collapsed them to "all OOS"; that grid was discarded and rerun. Weighting partially counters the label imbalance, since the headline metric averages per-label recall with CPI weights.
- Calibration: one scalar temperature per run, fitted on that run's validation rows by NLL (k-NN uses log vote shares with a 1e-4 floor). ECE is reported before and after.
- Encoder fine-tune: XLM-R base, same pools, class weights, early stopping and temperature scaling. Max 64 tokens, AdamW 3e-5, 6% warm-up, bf16 autocast.
- LLM: Anthropic SDK, one product per request, `claude-opus-5-5` (frontier) and `claude-haiku-5-5` (low-cost), adaptive thinking at effort `low`. The fixed system prompt holds instructions, the label list grouped by scope group with TPSO-grounded descriptions, and 2 official MOC *training* examples per label; it is prompt-cached (5,448 tokens). Structured output is a JSON schema with `group` then `label` (enums) and a verbalised `confidence` 0–100, which the coverage frontier uses.
  - No server-side refusal fallback: it would let another model answer and blur the per-model comparison. Refusals are recorded as abstentions.
  - Prices are from platform.claude.com/docs/en/about-claude/pricing on 2026-10-27, in USD per MTok (input / 5-min cache write / cache read / output): Opus 5.5 4 / 5 / 0.20 / 20; Sonnet 5.5 2 / 2.50 / 0.10 / 10; Haiku 5.5 0.10 / 0.125 / 0.01 / 0.50. Batch (−50%) is not used: real-time calls are simpler and the total spend is small.
  - Every response is cached under `.cache/llm/`, and every billed call is logged to `results/llm_costs.csv`. Each call first checks the logged total plus a worst-case estimate against the USD 60 cap.
  - Frontier model coverage: gold, Thai weak validation, and a fixed 1,500-item hash subsample of MOC test. The low-cost model covers all of MOC test.
- Expenditure-weighted accuracy: item weight ω_i = w_k · (1/π_i) / Σ_{j∈k} 1/π_j. Its weighted mean is Σ_k w_k r_k / Σ_k w_k, where r_k is the Horvitz–Thompson recall of label k over the Thai product universe. w_k is the price-updated CPI weight; leaves in a residual cell share it equally (primary) or by NNLS (sensitivity). OOS carries no CPI weight but counts in unweighted accuracy and macro-F1. Abstentions are wrong at full coverage and uncovered on the frontier.
- Cost model: API cost is the measured mean logged USD per call (cache writes included). Local compute is an *assumed* USD 1.00 per GPU-hour times measured wall-clock seconds per item (embedding plus prediction).
  - Encoding cost is timed separately from the corpus cache run: 5,000 sampled corpus texts, model load and one warm-up batch excluded, CUDA synchronised (`results/embed_timing.json`). A fine-tuned encoder's prediction time already includes its encoder.
- Cascade: a cheap classifier (e5-large MLP, bge-m3 MLP, or XLM-R, all on the full pool) keeps its prediction when its calibrated confidence is at least t; otherwise the product goes to the LLM. The dev sweep (`results/cascade_dev.csv`, MOC test and Thai weak validation) is used only to *choose* t and the cheap model. The chosen cascade is then scored once on gold, so the gold estimate is not tuned on gold.
- Gold evaluation (`cpilens eval`): every prediction file is scored with 1,000 stratified resamples, under equal and NNLS leaf weights. The outputs are `results/gold_scores.csv`, `gold_frontier.csv` (LLMs, cascades, cascade base models) and `gold_paired.csv`. Paired acc_w comparisons are fixed in code before any gold label exists:
  - Opus vs Haiku;
  - Haiku vs each cascade base model;
  - each cascade vs its LLM alone;
  - the brand/quantity ablations;
  - the cross-lingual transfer steps.
  - A cascade's threshold maximises Thai weak-validation acc_w (ties: cheaper). Its confidence is that of whichever model answered.

## Phase 4 — 2026-10-27 (materiality on gold: `cpilens econ`)
- Confusion counts per method come from the gold labels (UNK excluded). Universe shares of the true labels are Horvitz–Thompson estimates. Uncertainty comes from 1,000 Dirichlet posterior draws of the confusion matrix; the shares are held at their point estimates.
- Smoothing is empirical Bayes, worth α = 1 product per row. Row k's prior mean puts the method's pooled accuracy ρ on column k and spreads 1 − ρ over the other columns in proportion to where that method's errors go overall. A flat Dirichlet(α/C) prior was tried first and rejected. A sanity run scored one model against its own answers (a perfect classifier by construction) and showed about −13 bp headline YoY error, coming purely from flat mass sent to arbitrary items in sparse rows. Under the empirical-Bayes prior a perfect classifier gives exactly zero error in every draw (unit test).
- Variants:
  - `full`.
  - `coverage_only`: a perfect classifier on the gold universe. It is non-zero only if an item has no gold products and is renormalised within its group. The part due to classification is `full` minus `coverage_only`.
  - `noise_s0.05` / `noise_s0.10`: product-level sampling noise as a sensitivity. Each product's log price deviates from its item's by a random walk with an *assumed* annual s.d. σ. Item j averages max(1, N·inflow_j) products, with N the HT universe size. The estimate is the draw median.
- Flows in the decomposition are typed as within group (same 3-digit group), within division, cross division, OOS leak, or empty item. Outputs: `results/rq3_errors.csv`, `rq3_series.csv` (MoM/YoY official vs measured with draw bands) and `rq3_decomposition.csv`.

## Phase 5 — 2026-10-27 (figure style, gold-independent figures)
- `cpilens/viz/style.py`:
  - Libertinus Serif (SIL OFL, vendored in `figures/fonts/`), the house face used across the author's papers.
  - Okabe–Ito colours with one fixed slot per method family.
  - Full-width (16 cm), one-column (7.8 cm) and 16:9 slide sizes.
  - Light horizontal grid, no top or right spines.
  - Export to PDF, SVG (text as paths) and PNG at 300 dpi, with fixed metadata so reruns are byte-stable.
- The mission's figure conventions override the author's earlier house style (no in-figure titles). Every chart carries a title stating its message, a source line, and an optional note line. Titles and notes that contain a number build it from the results tables, never from literal text.
- Figure scripts read only `results/` tables. `cpilens eval` therefore also writes:
  - `gold_label_recall.csv`: HT recall, universe share and weight per label;
  - `gold_confusion_group.csv`: HT row-normalised group confusion;
  - `gold_cascade_sweep.csv`: every threshold on gold, descriptive only, with the dev-chosen one flagged.
- `cpilens econ` writes `rq3_flow_types.csv` (contribution sums by flow type) for all methods. It writes full decompositions (flows ≥ 0.01 bp) and monthly series only for the headline methods: both LLMs, the three cascade base models and their six cascades. Run settings are recorded in `rq3_meta.json` and `gold_summary.json`.
- Embedding map:
  - t-SNE (scikit-learn, fixed seed, PCA init, perplexity 30) of a balanced sample: ≤ 40 products per (label, Thai/other).
  - It is for display only. Alignment is measured in the full e5-large space: for up to 200 sampled Thai products per label, the share of their 10 nearest non-Thai neighbours that have the same label.
  - Labels with no non-Thai products have undefined alignment (null), not zero.
- Figure scripts were developed against a scratchpad run in which Opus's own gold answers stood in for the labels. Nothing from that run is written to `results/` or committed.
- All 12 mission figures have scripts (`figures/fig01`–`fig12`). Layout fixes made after viewing each one:
  - fig09 legend sits inside panel (a) (`loc="best"`); a figure-level legend collided with the title.
  - fig11 legend handles are explicit patches; empty `barh` calls drew the wrong colour.
  - Groups 421 and 721 are not in the published weights table (only their divisions are), so their display names come from a two-entry mapping in `cpilens/viz/names.groups()`.
- fig02 (coverage treemap): a squarified layout (`cpilens/viz/treemap.py`, no new dependency) of the `w_equal` leaf weights. The in-scope items are blue and each division's out-of-scope remainder is grey. The data are `results/coverage.csv`, written by `cpilens weights`. The in-scope item count shown in fig01 and fig02 is read from that table too.
- RQ4 (`cpilens agreement`, `cpilens/evaluate/agreement.py`):
  - Pass 1 is treated as the truth and pass 2 as a classifier. This puts the ceiling's `acc_w` on the same scale as every method's.
  - Agreement and Cohen's kappa are reported at item, group and division level, with 1,000 stratified resamples. Pairs with `UNK` in either pass are excluded and counted.
  - Disagreement is also given by division and by group of the pass-1 label.
  - Nearly all in-scope weight is in division 1, so a by-division split is nearly degenerate. fig12 therefore shows disagreement by level (panel a) and by group, for groups with n ≥ 5 (panel b). The by-division numbers stay in `rq4_by_class.csv`.
  - Spectrum's 8–10% human error rate (notes D6, p. 15) is compared in the text only, never plotted: it is a different taxonomy and design.

## Phase 6 — paper and publication (27 Oct 2026)
- Legends sit in a row under the chart, above the source line (`style.finish(legend=...)`), as in central-bank publications: they never cover data and never collide with panel titles.
- One axis wording for the headline metric everywhere (`style.ACC_LABEL`), in whole per cent; "basis points" in axis labels, "bp" after a number in a title.
- Method-family colours are reserved for methods: fig08 uses grey and blue for non-Thai and Thai products, fig11 the ordinal blue ramp for flow types.
- fig02 keeps a name strip above each division's cells and labels an item only when its name fits inside the cell; the rest are counted in the note and listed in the appendix table.
- `results/run_manifest.json` carries no timestamp or git hash, so `make paper` leaves the tree clean; the commit that holds the file is its version.
- `paper/main.pdf` is committed and linked from the README, as in the author's other repositories.
- The repository went public before the results exist (the author's decision after a review on their behalf); the README states the status and the release date. The v0.1.0 tag and release wait for Checkpoint C.
