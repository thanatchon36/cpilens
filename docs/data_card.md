# Data card: cpilens gold labels and product corpus

Counts below are not typed here; they are in `results/data_summary.json` (written by
`cpilens report`) and in the paper's Data section.

## What is released

| File | Content | Licence |
|---|---|---|
| `data/gold/sample.csv` | The gold sample: Thai-market Open Food Facts products with a Thai-script name, drawn by stratified sampling. Fields: `order` (pass-1 order), `barcode`, `source`, `stratum` (CPI weight cell of the weak label, or none), `incl_prob` (inclusion probability), `double` (in the second-pass subset), `name_th`, `name_en`, `brands`, `quantity`, `image_url`. | ODbL 1.0 |
| `data/gold/labels_pass1.csv`, `labels_pass2.csv` | One row per labelling action: `order`, `barcode`, `label` (CPI item code, `OOS` or `UNK`), `unsure`, `seconds`, `labelled_utc`. The latest row per product is the label; earlier rows are the audit trail. | ODbL 1.0 |
| `taxonomy/thcpi_scope.csv` | The in-scope CPI items, their COICOP 2018 class and a short description. | MIT |
| `taxonomy/off_categories_to_thcpi.csv` | Crosswalk from Open Food Facts category tags to CPI items (weak labels). | ODbL 1.0 (derived from the OFF taxonomy) |
| `taxonomy/thcpi_*weights*.csv` | Official aggregate weights and estimated item weights. Item weights inside an official weight cell are **estimates** (equal split; NNLS), not official figures. | MIT (derived from TPSO publications) |

The product corpus itself (`data/interim/`) is rebuilt from the public dumps by `make data`
and is not committed. Attribution for everything derived from Open Food Facts:
"© Open Food Facts contributors", <https://world.openfoodfacts.org>.

## How the gold labels were made

- **Frame:** products tagged as sold in Thailand with a name in Thai script.
- **Sample:** stratified by the CPI weight cell of the weak label, with a floor per
  stratum and the rest allocated half by stratum size and half by cell weight. Inclusion
  probabilities are stored; all gold estimates are design-weighted.
- **Labeller:** one person (the author), using the app in `cpilens/label/`. The app shows
  the product's names, brand, quantity and front image, and a search over item names. It
  never shows a model prediction, the weak label or the stratum.
- **Second pass:** a random subset relabelled blind, in a different order, at least 48
  hours after the first label. Agreement is intra-annotator, a lower bound on
  inter-annotator disagreement.
- **Special labels:** `OOS` (outside the in-scope items), `UNK` (cannot tell from the
  information shown; excluded from accuracy and counted), and an `unsure` flag.

## Known limitations

- Open Food Facts is crowd-sourced; its Thai products are not the assortment of Thai shops.
- One labeller; no adjudication.
- Product names and images are as entered by contributors and may be wrong.
- No prices: the data support classification, not index compilation.

## Out-of-scope uses

Not an official classification. Not to be used as, or to infer, any statistical office's
or central bank's internal data or methods.
