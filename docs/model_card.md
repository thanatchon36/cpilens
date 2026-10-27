# Model card: cpilens classifiers

Scores are not typed here: they are in `results/gold_scores.csv` and the paper.

## Task

Map a Thai (or mixed-language) product text, `name | brand | quantity`, to one of the
in-scope items of the Thai CPI (base 2566) or to `OOS`, with a calibrated confidence.

## Methods

| Family | Models | Trained on |
|---|---|---|
| Frozen embeddings + k-NN / logistic regression / MLP | `intfloat/multilingual-e5-large`, `BAAI/bge-m3` (open, run locally) | Weak labels from Open Food Facts tags (non-Thai and Thai) and official TPSO price-collection specifications |
| Fine-tuned encoder | `FacebookAI/xlm-roberta-base` | Same pools |
| LLM prompting | Claude Haiku 5.5, Claude Opus 5.5 (Anthropic API), few-shot, JSON-schema output | No training; official examples in the prompt |
| Cascade | A cheap classifier, routing low-confidence products to an LLM | Threshold chosen on development data |

Exact revisions, package versions and the seed are in `results/run_manifest.json`;
hyperparameters are in the paper's appendix (generated from the code).

## Evaluation

Expenditure-weighted accuracy (CPI-weighted mean item recall), accuracy over the held-out
list (design-weighted), macro-F1, accuracy–coverage and cost–accuracy frontiers, with
stratified bootstrap intervals on the reference set (held-out official specifications; see
the data card); and the error each method's confusion matrix induces in simulated headline,
food and core inflation, in basis points, under a stated product universe.

## Intended use

Research on the measurement error that product classification adds to price statistics.
A starting point for statistical offices or researchers building web or scanner price
indices for Thailand or other non-English markets.

## Not intended for

Producing official statistics without validation on the user's own data; any decision
about individuals or firms.

## Limitations

- The reference labels are the office's short price-collection specifications, not shop
  listings; two in-scope items have none and are not evaluated.
- No human-labelled product set and no label ceiling in this version.
- LLM behaviour and prices can change after the run date; responses are cached for
  reproducibility.
- Results are not directly comparable to the BIS Project Spectrum report (different
  taxonomy, language, data and metrics).
