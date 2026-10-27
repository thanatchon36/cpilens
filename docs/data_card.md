# Data card: the cpilens reference set and product corpus

Counts are not typed here; they are in `results/data_summary.json` (written by
`cpilens report`) and in the paper's Data section.

## What is released

| File | Content | Licence |
|---|---|---|
| `taxonomy/thcpi_scope.csv` | The in-scope CPI items, their COICOP 2018 class and a short description. | MIT |
| `taxonomy/off_categories_to_thcpi.csv` | Crosswalk from Open Food Facts category tags to CPI items (weak labels). | ODbL 1.0 (derived from the OFF taxonomy) |
| `taxonomy/thcpi_*weights*.csv` | Official aggregate weights and estimated item weights. Item weights inside an official weight cell are **estimates** (equal split; NNLS), not official figures. | MIT (derived from TPSO publications) |
| `data/gold/sample.csv` | A stratified sample of Thai-market Open Food Facts products drawn for hand labelling. **Not labelled and not used in this version**; kept for a later one. | ODbL 1.0 |

The product corpus itself (`data/interim/`) is rebuilt from the public dumps by `make data`
and is not committed. Attribution for everything derived from Open Food Facts:
"© Open Food Facts contributors", <https://world.openfoodfacts.org>.

## The reference set

- **Source:** the price-collection specifications the Ministry of Commerce publishes as open
  data, each under the official item code TPSO uses. They cover the whole basket, so
  specifications of items outside the scope carry the label `OOS`.
- **Split:** whole sub-items (seven-digit codes) are held out inside every item: a fifth of an
  item's sub-items (at least one) are test, another fifth validation, the rest training; an
  item with a single sub-item is split by specification. Test specifications whose text
  duplicates a training one are dropped.
- **Reference rows:** every held-out in-scope specification, plus a fixed hash-ordered sample
  of the held-out out-of-scope specifications carried with the matching design weight, so
  estimates refer to the whole held-out list.
- **Coverage:** two in-scope items (ready-to-eat food; sugar and confectionery) have no
  specifications in the published file and cannot be evaluated. Items with few specifications
  have wide intervals.
- **Use:** cascade thresholds and classifier settings are chosen on the training and
  validation splits only. The set was not sealed: the development grid logs its scores for
  every run, so they were visible during development.

## Labelling app (not used in this version)

`cpilens/label/` is a local app for labelling the product sample by hand: it shows names,
brand, quantity and the front image, searches item names in Thai and English, and never
shows a model prediction, the weak label or the stratum; it supports a blind second pass at
least 48 hours later. No labels were collected for v0.1.

## Known limitations

- Specifications are the office's own short descriptions, not shop or web listings.
- Open Food Facts is crowd-sourced; its Thai products are a development set only (weak
  labels from the crosswalk), never an evaluation set.
- No prices: the data support classification, not index compilation.

## Out-of-scope uses

Not an official classification. Not to be used as, or to infer, any statistical office's
or central bank's internal data or methods.
