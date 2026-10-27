# Project Spectrum — citable facts (with page numbers)

Source read in full on 2026-10-27:

> BIS Innovation Hub (2026): *Project Spectrum: using generative AI to enhance inflation nowcasting*,
> February. A collaboration between the BIS, the Deutsche Bundesbank and the ECB. 34 pages,
> ISBN 978-92-9259-929-4 (Online). Landing page <https://www.bis.org/publ/othp109.htm>
> (dated 17 February 2026). PDF:
> <https://www.bis.org/publications/project-spectrum-using-generative-ai-enhance-inflation-nowcasting.pdf>
> (SHA-256 `4cfe8d7f7e00e7aab21db1acd7f027072200816d62d58666312d4d400b4b7142`).

`https://www.bis.org/publ/othp109.pdf` returns an HTML page, not the PDF; use the URL above.
The report names no individual authors on its cover; participants are listed on p 31.
Cite it as an institutional report, not as a BIS Working Paper.

PDF page numbers equal the printed page numbers. **Rule:** the paper may state a fact about
Spectrum only if it appears below. Spectrum's numbers are never directly comparable to ours
(different data, taxonomy, languages, weights); say so wherever they sit side by side.

## Scale, cost and the core claim
| # | Fact | Page |
|---|---|---|
| S1 | The ECB Daily Price Dataset (DPD) holds billions of daily price-product observations for 34 million unique products. | 3 |
| S2 | Classifying the DPD with GPT-5 "would take over six months of computing time at a cost exceeding EUR 0.5 million" (executive summary). | 3 |
| S3 | The body text gives the same extrapolation as "over 200 days and … over EUR 650,000" for 34 million records; it is extrapolated from tests on a smaller number of records (fn 10). | 9 |
| S4 | AI was used only to turn product descriptions into text embeddings, then classic ML classified them; accuracy "comparable to LLM prompting"; the entire DPD was classified in five days for about EUR 1,500. | 3 |
| S5 | DPD collects about 4 million price-product observations per day; nearly a million new products may be added each month. | 7 |
| S6 | Direct prompting with the full category handbook can reach about 56,000 tokens per prompt. | 9 |
| S7 | Text embedding "eliminates the randomness associated with prompting". | 9 |
| S8 | Human annotators are assumed to label two products per minute, i.e. 4,800 per 40-hour week (fn 10). | 9 |

## Data and labels
| # | Fact | Page |
|---|---|---|
| D1 | Target taxonomy: ECOICOP 2018 v1, at the subclass (five-digit) level; ECOICOP 2018 v1 is not fully consistent with the final COICOP 2018 (fn 12). | 11 |
| D2 | The data set (as of December 2025) has ~34 million products, 111 shop identifiers (~60 retailers); names and descriptions average 60 and 581 characters; ~400,000 lack both. Prices are excluded from the classification data. | 13 |
| D3 | Reference (training) set: 30,000 products, deliberately not random — it overweights smaller categories. Test set: 30,000 products drawn at random. Both de-duplicated. | 13 |
| D4 | Reference-set labellers got iterative feedback; test-set labellers labelled once with no iteration. | 14 |
| D5 | Table 4 shows labeller disagreements (e.g. baby towel: "article for babies" vs "bathroom linen"; dry pasta from a food-delivery service: pasta vs take-away food). | 14 |
| D6 | **Label ceiling:** 250 labelled records were validated by additional expert annotators; human error rate estimated at **8–10%**; Fleiss' kappa described only as "moderate" at the five-digit level — **no numeric kappa is reported**. | 15 |
| D7 | The DPD covers ~50% of euro area CPI expenditure; the test set covers 154 of 253 categories, 50.16% of CPI (Eurostat euro area-19 weights, fn 15). | 15 |
| D8 | Web-scraped data have categorical gaps (energy, rents, telecoms, cars) and subclass imbalances relative to CPI weights. | 15 |

## Methods
| # | Fact | Page |
|---|---|---|
| M1 | Embeddings: OpenAI text-embedding-3-large (3,072 dims) and -small (1,536) were tried; all reported results use -large; accuracy is sensitive to the embedding model (tests not reported). | 18 |
| M2 | Input text = concatenation of name, description, shop category and sector. | 19 |
| M3 | KNN: odd k from 1 to 15; k = 11 had the best weighted-average F1 (by < 1 pp) but **k = 5 was chosen for better performance when accounting for CPI weights**; cosine distance; ties broken at random. | 19 |
| M4 | FFN: 85/15 stratified split; input 3,072 → hidden 128 → 64 → softmax over subclasses; Adam, cross-entropy, dropout, batch norm, weight decay; 50 epochs with early stopping (fn 16). | 20 |
| M5 | Direct prompting used GPT-5 with the ECOICOP 2018 v1 handbook; a rule was added to send food-delivery items to 11.1.1.2. | 21 |
| M6 | "Accuracy" at subclass level = in-class recall, TP/(TP+FN) (fn 17). | 22 |

## Results
| # | Fact | Page |
|---|---|---|
| R1 | Categories with at least 70% accuracy together cover 36% of CPI (KNN) (Graph 8). | 23 |
| R2 | At full coverage (~50% of CPI), weighted accuracy: direct LLM prompting 86%, FFN 80%, KNN 75% (Graph 9, p 25). | 24 |
| R3 | FFN beats KNN; KNN is more traceable/explainable. | 24, 27 |
| R4 | **"For inflation nowcasting, classification errors at the subclass (five-digit) level usually have a lower impact than those at higher hierarchy levels"**, depending on divergent price patterns across categories (e.g. olive oil vs butter). | 25 |
| R5 | Most predictions stay within the correct division (Graph 10, row-normalised division confusion). | 25–26 |
| R6 | Table 5: direct LLM ~500 s and EUR 22.2 per 1,000 products (printed "22,2"); embedding + KNN < 8.1 s and < EUR 0.030; embedding + FFN < 6.8 s and < EUR 0.031. | 26 |
| R7 | Production pipeline: ~3 hours of computing for one million new products. | 28 |
| R8 | Iterative refinement: batches of e.g. 1,000; 6,000 extra products labelled, split equally between reference and validation; reclassifying the whole DPD costs ~EUR 150 (fn 19). | 29 |

## Next steps Spectrum names (our positioning hooks)
| # | Fact | Page |
|---|---|---|
| N1 | "A potential next step is to test the solution with different data sets and languages." | 30 |
| N2 | A further validation step is to construct CPI indices at the ECOICOP subclass level and back-test them against official inflation; this needs historical price data and is a priority for the follow-up phase. | 30 |

**Positioning:** cpilens takes up N1 (Thai and other Asian languages). On N2 it does **not**
back-test web-price indices. It complements N2 by quantifying R4: given a confusion matrix and
official weights and indices, how large is the induced inflation error? Describe it as a
complement, never as "doing N2".

## Things not to cite (internal inconsistencies or ambiguity in the PDF)
- The Graph 6 caption names the predicted class "01.2.3 Fruit juices", while the text says 01.2.2.3 (pp 19–20).
- The cover reads "enchance"; Graph 8's panel label reads "FNN".
- Graph 7 has no numeric per-category values; do not read numbers off the treemaps.
