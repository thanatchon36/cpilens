<p align="center"><img src="figures/out/fig10_bp_error.png" alt="RMSE of year-on-year inflation error by method, headline, food and core" width="860"></p>

**In five sentences.** Using public data only, we classify Thai product text into the items of
the Thai CPI and measure what each method's errors would do to measured inflation, scoring on
{NRefSpecs} held-out official price-collection specifications weighted by CPI expenditure.
Transfer from non-Thai products alone gets an {NameEfive} classifier to {AccWEfiveMLPZeroShot}%
of the evaluated CPI weight, and all {NMocTrainval} official labels raise it to {AccWEfiveMLP}%.
{NameOpus} reaches {AccWOpus}% and {NameHaiku} {AccWHaiku}%, and a cascade from {NameEfive} to
{NameOpus} matches {NameOpus} at {CostPctCasEfiveMLPOpus}% of its cost. With {NameOpus}'s
confusion matrix, headline year-on-year inflation would be misstated by
{BpYoyRmseHeadlineOpus} basis points (RMSE; {CfgCI}% interval {BpYoyRmseHeadlineOpusLo}–{BpYoyRmseHeadlineOpusHi}),
against {BpYoyRmseHeadlineEfiveMLP} for the {NameEfive} classifier. Accuracy does not rank
methods by this error, and most of the LLMs' remaining errors are gaps in the documentation of
the official basket, not misreadings of Thai.

*Numbers in this block are filled in from `results/` by `cpilens report` (template:
`docs/readme_results.md`). Total LLM spend for the study: USD {LLMSpendUSD}.*
