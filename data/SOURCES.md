# Data sources

Every source used by cpilens, with URL, retrieval date (UTC), licence and checksum. Raw files live
in `data/raw/` and are not committed (size or licence); `make data` re-downloads them.
Licences of the Thai government sources were checked on 2026-10-27. None of the four source
pages states a licence or terms of use. Related TPSO datasets on data.go.th and
catalog-tpso.moc.go.th are listed variously as Open Data Common or Creative Commons
(Attribution, Share-Alike, Non-Commercial). This repository therefore treats them under the
most restrictive of those terms: attribution, non-commercial research use, and **no
redistribution of the raw files**. They are downloaded by `cpilens fetch-raw` and are not
committed; the repository holds only specification ids, official item codes and numbers
derived from them.

| ID | Source | URL | Retrieved | Licence | Notes |
|---|---|---|---|---|---|
| OFF | Open Food Facts product database, Parquet export (`food.parquet`, 7.996 GB, 4,837,301 products) | <https://huggingface.co/datasets/openfoodfacts/product-database> (revision `3ac7e8b6e937…`) | 2026-10-27T04:33Z–04:42Z | ODbL 1.0 (database); DbCL (contents); images CC BY-SA | SHA-256 `f94a5a16…b207b0a7`. Attribution: "© Open Food Facts contributors". Derived labelled data must be released under ODbL. |
| OFF-TAX | Open Food Facts category taxonomy (`categories.json`, 14,774 nodes) | <https://static.openfoodfacts.org/data/taxonomies/categories.json> | 2026-10-27 | ODbL 1.0 | Used for tag ancestors in the silver-label crosswalk. SHA-256 `eb03aefa…fe2d1020d5`. |
| OBF | Open Beauty Facts Parquet export (`beauty.parquet`, 65.8 MB) | same dataset, `beauty.parquet` | 2026-10-27T04:33Z | ODbL 1.0 | SHA-256 `ec9f13d2…8acf00db`. Only 86 products tagged Thailand (API count) — too few for the scope. |
| OPF | Open Products Facts | <https://world.openproductsfacts.org> | 2026-10-27 (API count only) | ODbL 1.0 | 34 products tagged Thailand. Not used. |
| TPSO-API | TPSO Open API, national monthly CPI (`POST /OpenApi/Cpig/Month`, `GET /OpenApi/Cpig/Month/MasterData`), base 2566 = 100, type `TG` | <https://index-api.tpso.go.th> (docs: <https://index.tpso.go.th/document/cpig>) | 2026-10-27T04:41Z | Thai government open data; no licence stated (checked 2026-10-27), see above | 194 index codes; data from 2519 to 9/2569; 93 monthly files 1/2562–9/2569 pulled. Indices only — no weights. |
| TPSO-PR | TPSO monthly CPI release, national: 21 releases, Jan 2568 to Sep 2569 (`Cpig012568_tg.pdf` … `Cpig092569_tg.pdf`; every file with its SHA-256 in `data/manifest.csv`) | <https://uploads.tpso.go.th/economic/pdf/Cpig092569_tg.pdf> | 2026-10-27 | Thai government publication; no licence stated (checked 2026-10-27), see above | p 11 table: price-updated weight shares for 29 aggregates. SHA-256 `a308fe86…` (Sep), `9da594f5…` (Aug). |
| TPSO-NOTE | TPSO, "คำชี้แจง การปรับปรุงดัชนีราคาผู้บริโภค ปีฐาน 2566" (Feb 2568) | <https://www.price.moc.go.th/price/fileuploader/file_cpi/cpi_note_2566.pdf> | 2026-10-27 | Thai government publication; no licence stated (checked 2026-10-27), see above | Official base-2566 division weights; 464 items (430 in base 2562); in use from Jan 2568; core = 338 items. SHA-256 `cab38b23…`. |
| MOC-SPEC | Ministry of Commerce open data, CPI product specifications (`/OpenData/getCPIProducts`) | <https://data.moc.go.th/OpenData/CPIIndexes> | 2026-10-27 | Thai government open data; no licence stated (checked 2026-10-27), see above | 34,766 records: 609 item headers and 34,157 TPSO price-collection specifications (Thai text with brand and size) under 16-digit item codes, including items from earlier bases. SHA-256 `ff08b6ed…`. |
| UNSD-COICOP | United Nations Statistics Division (2018), *Classification of Individual Consumption According to Purpose (COICOP) 2018*, pre-edited white-cover version | <https://unstats.un.org/unsd/classifications/unsdclassifications/COICOP_2018_-_pre-edited_white_cover_version_-_2018-12-26.pdf> | 2026-10-27 | © United Nations; class codes and titles cited | Class codes in `taxonomy/thcpi_scope.csv` come from the class list on pp 43–52. SHA-256 `3ce18c1b…99615f4bff4`. |
| SPECTRUM | BIS Innovation Hub (2026), Project Spectrum report | <https://www.bis.org/publications/project-spectrum-using-generative-ai-enhance-inflation-nowcasting.pdf> | 2026-10-27 | © BIS 2026, all rights reserved; limited extracts with source stated | Not redistributed. SHA-256 `4cfe8d7f…`. See `docs/spectrum_notes.md`. |

Not used, by rule: retailer websites (no scraping without explicit approval), secondary
aggregators (CEIC, Moody's/economy.com, Trading Economics), any Bank of Thailand non-public data.
