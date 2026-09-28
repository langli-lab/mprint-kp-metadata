# MPRINT Knowledge Portal Metadata

This repository holds the metadata standards and model resources behind the MPRINT (Maternal and Pediatric Precision in Therapeutics) Knowledge Portal. It contains:

- curation schemas that define which variables are extracted from published articles, and how;
- reference vocabularies for tagging drugs and maternal / pediatric subpopulations;
- configuration, tokenizer files, and release instructions for six BioBERT classifiers.


## Contents

| Path | What it is |
|---|---|
| `PK_curation_variables.xlsx` | Data dictionary for pharmacokinetic (PK) article curation |
| `PE:CT Curation/PE-CT curation information.docx` | Extraction manual for pharmacoepidemiology (PE) and clinical trial (CT) articles |
| `PE:CT Curation/Data curation template for PE-CT studies.xlsx` | Blank PE/CT curation template |
| `subpopulation_keyword.csv` | Keywords and MeSH terms mapped to 12 maternal / pediatric subpopulations |
| `drug_dictionaryI.csv` | Drug names mapped to UMLS concept IDs (CUIs) and MeSH terms |
| [`6_models_MPRINT/`](6_models_MPRINT/README.md) | Six BioBERT checkpoints: documentation, configuration, tokenizers, and original example code; weights are packaged separately for release |

## BioBERT models

The six supplied checkpoints are Biomarker, CT, FBNSTP, PE, PK, and VC. Their large
weight files are kept out of Git and packaged as individual GitHub Release
downloads. See the [model README](6_models_MPRINT/README.md) for release status,
download instructions, checkpoint details, and publishing steps.

## Curation schemas

### PK studies: `PK_curation_variables.xlsx`

Each sheet defines one domain. Each row is a variable, with a `Curation_Variable` name and a `Description`. Every domain has two sheets:

- **summary data**, for group-level results as reported in the paper;
- **individual data**, for per-patient results, keyed by `Patient ID`.

| Domain | Summary sheet | Individual sheet | Covers |
|---|---|---|---|
| Population | 17 variables | 23 variables | Demographics, comorbidities, pregnancy stage, gestational age |
| Drug | 20 variables | 22 variables | Dose, route, frequency, schedule |
| Specimen | 17 variables | 21 variables | Specimen type (plasma, breast milk, cord blood, etc.), time from dose to sample |
| PK param | 21 variables | 37 variables | Analyte, parameter (AUC, Cmax, clearance, etc.), value, time |
| Outcomes | 17 variables | 20 variables | PD, clinical and pharmacoepidemiology outcomes |

The sheet types share these conventions:

- **Summary sheets** use the same statistic block: `Value`, `Unit`, `Summary Statistics`, `Variation type` / `Variation value`, `Interval type` / `Interval low` / `Interval high`, and `Subjects n`.
- **Individual sheets** use repeating groups (`Characteristic 1`–`6` with `Value` and `Unit`, or `Parameter 1`–`5` with `Drug name`, `Time`, `Value` and `Unit`).
- **Most sheets** repeat `Population`, `Pregnancy Stage` and `Gestational Age`, so each value can be tied to a subpopulation.

### PE and CT studies: `PE:CT Curation/`

The manual (`PE-CT curation information.docx`) defines every column. For each column it gives where to look in the article, keyword sets, the PE versus CT interpretation, normalization rules, and valid and invalid examples. The template workbook has two sheets:

| Sheet | Granularity | Columns |
|---|---|---|
| Study info | One row per paper | PMID, Population, Study type, Study design, Pregnancy Stage, Drug names, Data source, Inclusion, Exclusion, Outcomes, Sample size |
| Outcomes | One row per reported result | PMID, Characteristic/risk factor, Exposure, Outcomes, Statistic, Value, Unit, Variability statistic, Variability value, Interval type, Interval low, Interval high, P-value, Notes |

The main rules from the manual are:

- **Study type** is `PE` (the drug exposure is observed) or `CT` (investigators assign the intervention). Labels such as PK, PD or safety can be added only after PE or CT has been chosen.
- **Extract only what the paper reports.** Values are stored as strings, with no calculation, conversion or rounding. Missing values are `N/A`, and a reported zero stays `0`.
- **Keep results separate.** Arms, subgroups, models, time points and analysis sets each get their own row.

## Population definitions

The PE/CT manual defines the vocabulary for the `Population` and `Pregnancy Stage` fields:

| Maternal pregnancy stage | Definition |
|---|---|
| Trimester 1 | ≤ 14 weeks of pregnancy |
| Trimester 2 | 15–28 weeks of pregnancy |
| Trimester 3 | ≥ 28 weeks of pregnancy |
| Fetus / fetal stage | Baby during pregnancy |
| Parturition / labor / delivery | The process of giving birth |
| Postpartum | 6–8 weeks (about 2 months) after birth |
| Nursing / breastfeeding / lactation | |

| Pediatric age group | Definition |
|---|---|
| Preterm / premature | ≤ 37 weeks of gestation |
| Neonates / newborns | Birth to 1 month |
| Infants | 1 month to 1 year |
| Children | 1 year through 12 years |
| Adolescents | 13 years through 17 years |
| Adults | 18 years or older |

## Reference vocabularies

### `subpopulation_keyword.csv`

This file maps 171 terms to a population and one of 12 subpopulations for siver version of the MPRINT Knowledge Portal. It has one row per term, with 171 rows in total.:

| population | subpopulations |
|---|---|
| maternal | maternal, pregnancy, labor, lactation, postpartum |
| pediatric | infant, pediatric, premature, newborn, child, fetal, neonate |

| Column | Description |
|---|---|
| term | The keyword or MeSH heading |
| term_type | `keyword` (free-text variant, e.g. `breastfed`) or `mesh` (MeSH descriptor) |
| population | `maternal` or `pediatric` |
| subpopulation | One of the 12 labels above |
| canonical_term | The normalized form: the canonical keyword for free-text variants, or the official MeSH heading for MeSH terms |
| mesh_heading, mesh_ui | The MeSH descriptor the term maps to, and its ID (e.g. `D011247` for Pregnancy) |
| also_indicates | The other population, when a term involves both mother and child (e.g. `breastfeeding`, `in utero`) |


When you use the file for text matching:

- **Match whole words only.** Substring search finds `child` inside `childbirth` and `labor` inside `laboratory`.
- **Expect false positives in drug literature.** Common ones are `child` in Child-Pugh, `delivery` in drug delivery, `fetal` in fetal bovine serum, `nursing` in nursing home, and `weaning` from a ventilator. Confirm these with a second term.
- **Match MeSH rows on `mesh_heading` or `mesh_ui`.** A few source terms are capitalized differently from MeSH (e.g. `Fertilization In Vitro` vs. `Fertilization in Vitro`).

### `drug_dictionaryI.csv`

This file has one row per drug name and UMLS CUI pair: 7,647 rows, covering 7,030 drug names and 7,630 CUIs.

| Column | Description |
|---|---|
| drug_name | The drug name as found in the literature (generic, brand, chemical or class) |
| cui | The UMLS concept unique identifier |
| mesh_term | A MeSH heading linked to the drug |
| papers | The number of papers for the drug–MeSH pair |
