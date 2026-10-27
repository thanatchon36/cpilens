# Verified references

`paper/refs.bib` holds 33 entries. Each was checked on 2026-10-27 against the source in the third
column: Crossref or doi.org for DOIs, the publisher, proceedings or journal page, arXiv's API, or
the issuing institution's own page or PDF. No field was filled from memory. All 15 DOIs resolve
at doi.org. The file compiles with `natbib` + `plainnat` without BibTeX warnings.

Keys already cited in `paper/sections/*.tex` (at the time of writing): `bis2026spectrum`,
`chen2024bgem3`, `cohen1960kappa`, `conneau2020xlmr`, `efron1993bootstrap`,
`geifman2017selective`, `guo2017calibration`, `horvitz1952`, `imf2020cpimanual`,
`moc2026specs`, `off2026`, `tpso2025cpinote`, `tpso2026api`, `unsd2018coicop`, `wang2024me5`.
They all exist in `refs.bib`.

## Positioning

| bibkey | What it supports in the paper | Where verified | Uncertain |
|---|---|---|---|
| `bis2026spectrum` | The benchmark: embeddings + classic ML vs LLM prompting for ECOICOP classification of ECB web-scraped data | <https://www.bis.org/publ/othp109.htm> (page title checked); details in `docs/spectrum_notes.md` | Uses the corporate author `{BIS Innovation Hub}` so natbib prints "BIS Innovation Hub (2026)". No personal authors. |

## Online and web-scraped prices

| bibkey | What it supports | Where verified | Uncertain |
|---|---|---|---|
| `cavallo2016bpp` | Online prices as a measurement source (Billion Prices Project) | Crossref, doi:10.1257/jep.30.2.151 (JEP 30(2):151–178) | — |
| `cavallo2017online` | Online and offline prices of multi-channel retailers are largely the same, so web text/prices are a valid frame | Crossref, doi:10.1257/aer.20160542 (AER 107(1):283–303) | Full title has the subtitle "Evidence from Large Multi-Channel Retailers". |
| `aparicio2020forecasting` | Online prices improve inflation forecasts | Crossref, doi:10.1016/j.ijforecast.2019.04.018 (IJF 36(2):232–247) | — |
| `macias2023nowcasting` | Nowcasting food inflation from a large set of scraped online prices | Crossref, doi:10.1016/j.ijforecast.2022.02.007 (IJF 39(2):809–826) | The Crossref record lists no affiliations. Don't call it a central-bank study unless the article is checked. |

## Scanner and web data in official CPIs; automated product classification

| bibkey | What it supports | Where verified | Uncertain |
|---|---|---|---|
| `eurostat2017scanner` | Eurostat's practical guide on supermarket scanner data in the HICP | PDF from the CIRCABC link on <https://ec.europa.eu/eurostat/web/hicp/publications>: cover "Practical Guide for Processing Supermarket Scanner Data", Eurostat, September 2017 | The URL is the HICP publications page (the PDF lives on CIRCABC behind an opaque link). |
| `eurostat2020webscraping` | Eurostat's web-scraping guidelines for the HICP; section 5 is "Classification and data validation" | PDF at the URL in the entry: cover "Practical guidelines on web scraping for the HICP", November 2020 | — |
| `eiglsperger2019hicp` | The ECB/HICP moved to more scanner data and web scraping | ECB page: "Prepared by Martin Eiglsperger. Published as part of the ECB Economic Bulletin, Issue 2/2019" | Box number not stated on the page (the URL suggests box 4), so it is left out. |
| `metcalfe2016clip` | ONS: ML (clustering + decision tree) to group web-scraped products into price indices | ONS article page: title, authors, 30 November 2016 | — |
| `martindale2020semi` | ONS: semi-supervised ML (label propagation) to classify web-scraped products into CPIH basket items | Crossref, doi:10.1017/dap.2020.13 (Data & Policy 2, e12) | — |
| `harms2019cbs` | Statistics Netherlands: ML classification of products into CPI categories (logistic regression best) | CBS PDF: cover, authors, "Statistics Netherlands, Discussion Paper, November 2019" | — |
| `roberson2021applying` | NLP classification of UPC product descriptions into a product classification (NAPCS), >90% accuracy | Crossref, doi:10.2478/jos-2021-0017 (JOS 37(2):395–410) | **The author is at the U.S. Census Bureau, not Statistics Canada**, and it is the Economic Census, not a CPI. Cite it as a statistical-office product-coding example, not a CPI one. |
| `abs2020webscraping` | ABS uses web-scraped prices in the CPI and is "investigating the use of machine learning to automate the classification" | ABS article page, released 29 April 2020 | — |
| `unece2021ml` | International survey of ML in official statistics, including a coding and classification theme and an HICP product-classification pilot | PDF title page "United Nations, Geneva, 2021"; Crossref, doi:10.18356/9789210011143 | Crossref/UN iLibrary dates the record 21 March 2022; the title page says 2021. Year 2021 used. |

## Index compilation and classification standards

| bibkey | What it supports | Where verified | Uncertain |
|---|---|---|---|
| `imf2020cpimanual` | The Lowe index and price-updated weights; CPI compilation | IMF PDF (<https://www.imf.org/-/media/files/data/cpi/cpi-manual-concepts-and-methods.pdf>): cover, CIP data ("Washington, DC: International Monetary Fund, [2020]", ISBN 978-1-48435-484-1); text mentions "Lowe index" 59 times. DOI from Crossref | Crossref's record lists one personal author ("Graf, Brian"), which is wrong for this joint manual. The entry uses the six organisations on the cover instead. |
| `unsd2018coicop` | COICOP 2018 class codes in `taxonomy/thcpi_scope.csv` (taken from pp 43–52 of this version) | PDF at the URL in the entry: cover "ST/ESA/STAT/SER.M/99 … White cover publication, pre-edited text … United Nations, New York, 2018". SHA-256 matches `data/SOURCES.md` | This is the pre-edited version the repo actually used. Page numbers differ from the final edition. |
| `unsd2023coicop` | The final, officially edited COICOP 2018 publication | PDF `COICOP_2018_draft_publication.pdf` on unstats.un.org: cover "… New York, 2023"; ISBN from <https://unstats.un.org/unsd/classifications/Family/Detail/2094> | Not cited yet. Use it if the paper wants the final edition; keep `unsd2018coicop` for page references to the white-cover text. |

## Thai CPI and product data

| bibkey | What it supports | Where verified | Uncertain |
|---|---|---|---|
| `tpso2025cpinote` | Base year 2566 (2023), 464 items, rebasing history, division weights | PDF at the URL in the entry: heading "คำชี้แจง การปรับปรุงดัชนีราคาผู้บริโภค ปีฐาน 2566", TPSO / Ministry of Commerce; PDF created 11 Feb 2025 | The English title is a translation. The year/month (Feb 2025) comes from the PDF metadata and the text ("ในปี 2568…"), not from a printed date. |
| `tpso2026api` | Monthly CPI indices from the TPSO Open API | <https://index.tpso.go.th/document/cpig> ("API Document", endpoints for monthly, quarterly and yearly data); <https://index-api.tpso.go.th/OpenApi/Cpig/Month/MasterData> returned JSON | The title is descriptive. The page has no formal title beyond "API Document". |
| `moc2026specs` | TPSO price-collection specifications used as official labels | <https://data.moc.go.th/OpenData/CPIIndexes>: page title "ราคาสินค้าขายปลีก - ดัชนีราคา - ข้อมูลเปิดเผยของกระทรวงพาณิชย์ MOC Open Data"; the page mentions `getCPIProducts`; the description says TPSO compiles the data | The English title is a translation. The bare endpoint `https://dataapi.moc.go.th/getCPIProducts` returned 404 without parameters, so it is not cited. |
| `off2026` | Open Food Facts product text | <https://world.openfoodfacts.org/data> (200); citation form from <https://world.openfoodfacts.org/scientific-publications> ("Open Food Facts Database [Data set]", give snapshot date); Hugging Face dataset page (200) | No peer-reviewed paper describes the database, so this is a `@misc`. The Hugging Face revision hash comes from `data/SOURCES.md`. Open Beauty Facts is covered in the text, not as a separate entry. |

## Multilingual encoders and embeddings

| bibkey | What it supports | Where verified | Uncertain |
|---|---|---|---|
| `conneau2020xlmr` | XLM-R encoder (fine-tuned baseline) | ACL Anthology BibTeX (2020.acl-main.747) and Crossref | — |
| `wang2024me5` | multilingual-E5 embeddings | arXiv API: 2402.05672, "Multilingual E5 Text Embeddings: A Technical Report", 6 authors, 8 Feb 2024 | arXiv only. No journal reference on arXiv. |
| `chen2024bgem3` | BGE-M3 embeddings | ACL Anthology BibTeX (2024.findings-acl.137) and Crossref; also arXiv 2402.03216 | First author is "Jianlyu" in ACL/Crossref but "Jianlv" on arXiv. The ACL form is used. |
| `reimers2019sbert` | Sentence embeddings (optional) | ACL Anthology BibTeX (D19-1410) | **Pages:** ACL Anthology says 3982–3992 and Crossref says 3980–3990. The Anthology is used. |

## Calibration, selective classification, visualisation

| bibkey | What it supports | Where verified | Uncertain |
|---|---|---|---|
| `guo2017calibration` | Temperature scaling | PMLR page and BibTeX (v70, 1321–1330) | — |
| `geifman2017selective` | Accuracy–coverage frontier / selective classification | NeurIPS proceedings page and official `.bib` (NIPS 30) | The official `.bib` has empty pages. A secondary listing gives 4878–4887, which is unverified, so pages are left out. |
| `vandermaaten2008tsne` | t-SNE embedding plots | JMLR page and official `.bib` (9(86):2579–2605) | — |

## LLMs for annotation and cascades

| bibkey | What it supports | Where verified | Uncertain |
|---|---|---|---|
| `gilardi2023chatgpt` | LLMs as annotators/classifiers vs humans | Crossref, doi:10.1073/pnas.2305016120 (PNAS 120(30), e2305016120) | — |
| `chen2023frugalgpt` | LLM cascade: a cheap model first, an expensive model on hard items | arXiv API: 2305.05176 (9 May 2023) | arXiv only. I couldn't confirm a later peer-reviewed version (e.g. TMLR). |

## Survey sampling, agreement, resampling

| bibkey | What it supports | Where verified | Uncertain |
|---|---|---|---|
| `horvitz1952` | Horvitz–Thompson weighting of the stratified gold sample | Crossref, doi:10.1080/01621459.1952.10483446 (JASA 47(260):663–685) | — |
| `cohen1960kappa` | Cohen's kappa for inter-annotator agreement | Crossref, doi:10.1177/001316446002000104 (EPM 20(1):37–46) | — |
| `efron1993bootstrap` | Bootstrap confidence intervals | Open Library record from LoC MARC (Chapman & Hall, New York, 1993; series no. 57; LCCN 93004489); DOI from Crossref | The DOI points to the CRC/Taylor & Francis e-book, which Crossref dates 1994. Year 1993 (first printing) is kept. |

## Skipped candidates

| Candidate | Why skipped |
|---|---|
| "Myklatun", Statistics Norway, ML classification into COICOP | No author by that name found in searches. The closest SSB-related item is a 2021 NMBU master's thesis (D. M. Müller, "Classification of consumer goods into 5-digit COICOP 2018 codes"), a thesis rather than a statistical-office publication. Not added. |
| Statistics Canada paper on ML classification of scanner data for the CPI | Not found as a citable paper. A StatCan scanner-data SVM project appears only in a 2018 arXiv survey (Beck et al., arXiv:1812.10422) and conference slides. The Laroche & Tremblay paper is about the Retail Commodity Survey (NAPCS), not the CPI. Not verified enough to add. |
| ECB Daily Price Dataset (DPD) | No standalone ECB publication found. Cite it through `bis2026spectrum` (pp 3, 7, 13). |
| ESSnet Big Data / Web Intelligence Network deliverables on web-scraped prices | No stable, citable deliverable located in the time available. `eurostat2020webscraping` covers the same ground with an official source. |
| Eurostat, *Guide on multilateral methods in the HICP*, 2022 edition (KS-GQ-21-020, doi:10.2785/873932) | Verified but left out because it is about index methods for scanner data, not classification. Add it if needed. |
| Eurostat, *HICP Methodological Manual*, 2024 edition | Landing page seen, but the bibliographic details were not checked. Not added. |
| Statistics Austria (Boettcher et al., NTTS 2021) ML classification of transaction/web data | Only an abstract was seen. Not added. |
| FrugalGPT peer-reviewed version | Not confirmed. The arXiv version is cited instead. |
