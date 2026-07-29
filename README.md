# FaireduPlus

Enhancing Intersectional Fairness in Educational Machine Learning through Subgroup-Aware Synthetic Data Generation

**Authors:**
- Nga Pham¹
- Minh Kha Do²
- Quang Trung Doan³
- Anh Nguyen-Duc⁴
- Pham Ngoc Hung⁵

¹ Dainam University, Hanoi, Vietnam
² Latrobe University, Australia
³ FPT University, FPT Polytechnic, Hanoi, Vietnam
⁴ University of South Eastern Norway, Bø I Telemark, Norway
⁵ VNU University of Engineering and Technology, Hanoi, Vietnam

## Description

FaireduPlus studies **intersectional fairness** (fairness across combinations of protected
attributes, e.g. gender × disability) in ML models trained on educational data. It combines two
stages:

1. **Synthetic data generation** ([generate_sdg.py](generate_sdg.py)) — balances underrepresented
   intersectional subgroups by generating synthetic rows with either an LLM (via
   [sdgx](https://github.com/hitsz-ids/synthetic-data-generator)'s `SingleTableGPTModel`) or
   [CTGAN](https://github.com/sdv-dev/CTGAN).
2. **Bias mitigation and evaluation** ([fairedu.py](fairedu.py)) — implements the **FairEDU**
   mitigation algorithm, which decorrelates non-protected features from protected attributes via
   linear regression before model training, then compares fairness/performance metrics
   before vs. after mitigation across five classifiers (logistic regression, random forest,
   gradient boosting, decision tree, MLP), using [aif360](https://github.com/Trusted-AI/AIF360)
   metrics (Disparate Impact, Statistical Parity Difference, Average/Equal Odds Difference, etc.).

## Dataset Information

Preprocessed datasets used in this project are hosted at Zenodo:
https://zenodo.org/records/17933909

Test data is expected at `<dataset_folder>/<dataset_name>/test_<dataset_name>.csv`, and each
dataset folder contains subgroup CSV splits (e.g. `Gender_1_Debtor_0_Probability_1.csv`) used for
synthetic generation.

Supported datasets (`--dataset-name`):

| Name | Protected attributes | Description |
|---|---|---|
| `student_dropout` | `Gender`, `Debtor` | Predicting student dropout/academic success |
| `student_oulad` | `gender`, `disability` | Open University Learning Analytics Dataset (OULAD) |
| `student_performance` | `sex`, `health` | Secondary school student performance (SP) |
| `DNU` | `gender`, `age`, `birthplace` | Da Nang University student outcomes dataset |

### Third-party data sources

- **Student Performance (SP)**: UCI Machine Learning Repository —
  https://archive.ics.uci.edu/dataset/320/student+performance
- **Student Dropout (SD)**: UCI Machine Learning Repository, "Predict Students' Dropout and
  Academic Success" — https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- **OULAD**: Open University Learning Analytics Dataset —
  https://analyse.kmi.open.ac.uk/open-dataset
- **DNU**: privately collected student records from Da Nang University; not publicly redistributed.

The CSV splits and merged files hosted on Zenodo above are preprocessed derivatives of the SP, SD,
and OULAD sources cited. Refer to each original source for licensing terms before redistribution.

## Code Information

- [fairedu_plus.py](fairedu_plus.py) — CLI entry point; orchestrates generation + evaluation.
- [generate_sdg.py](generate_sdg.py) — synthetic data generation (LLM or CTGAN) per intersectional
  subgroup, and merging of generated splits.
- [fairedu.py](fairedu.py) — dataset preprocessing, the FairEDU mitigation algorithm, model
  training/evaluation, and results export (`run_fairedu`, `save_results_to_file`).
- [utils.py](utils.py) — CSV merge/split helpers and dataset-specific column definitions.

## Requirements

- Python 3.9+
- Packages: `pandas`, `numpy`, `scikit-learn`, `aif360`, `ctgan`, `sdgx`, `statsmodels`
- For LLM generation, set `OPENAI_API_KEY` (used by `generate_sdg.generate_by_llm`).

Install dependencies:
```bash
pip install pandas numpy scikit-learn aif360 ctgan sdgx statsmodels
```

## Usage Instructions

1. Download the dataset(s) from the Zenodo link above and place them under a local
   `<dataset_folder>/<dataset_name>/` directory.
2. (Optional, for LLM generation) export `OPENAI_API_KEY=<your key>`.
3. Run the pipeline:

```bash
python fairedu_plus.py \
  --dataset-name student_dropout \
  --dataset-folder /path/to/original_dataset \
  --generator LLM \
  --merged-output-file-name merged_output.csv \
  --seed 42
```

This generates synthetic data to balance intersectional subgroups, merges it with the real
training data, runs FairEDU mitigation, and writes a results file
(`results_<dataset_name>.csv`) next to the merged training file with before/after fairness and
performance metrics.

### CLI options (`fairedu_plus.py`)
- `--dataset-name` (`student_dropout`, `student_oulad`, `student_performance`, `DNU`)
- `--dataset-folder` path to dataset root used to locate the test CSV
- `--generator` choose `LLM` or `CTGAN`
- `--merged-output-file-name` name for merged synthetic output
- `--run-splitted-file` / `--no-run-splitted-file` choose split vs combined training files
- `--seed` random seed for reproducibility

### Additional examples

CTGAN without split files:
```bash
python fairedu_plus.py --dataset-name student_dropout --generator CTGAN --no-run-splitted-file
```

OULAD dataset with explicit dataset folder:
```bash
python fairedu_plus.py --dataset-name student_oulad --dataset-folder ./dataset
```

## Methodology

1. **Preprocessing** — dataset-specific cleaning/encoding in `run_fairedu` (drop nulls, encode
   categoricals, scale numeric features).
2. **Baseline evaluation** — train each classifier on the original data; record accuracy,
   precision, recall, F1, and fairness metrics per protected attribute.
3. **FairEDU mitigation** — for each non-protected feature, fit an OLS regression against the
   protected attributes; if the relationship is statistically significant (p < 0.05), subtract the
   fitted linear effect from the feature to decorrelate it, then drop the protected attributes.
4. **Post-mitigation evaluation** — retrain each classifier on the decorrelated data and record the
   same metrics for comparison.
5. **Export** — before/after rows per classifier are written to CSV/Excel/pickle
   (`save_results_to_file`).

## Citations

If you use this work, please cite the SSRN preprint:
```
SSRN Scholarly Paper 5290738. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5290738
```

## Acknowledgements

This work builds upon the following open-source projects:
- https://github.com/hitsz-ids/synthetic-data-generator
- https://github.com/fairnesstest/LTDD

## License & Contribution Guidelines

No license file is currently included in this repository; contact the authors before reuse or
redistribution. Contributions and issue reports are welcome via pull request.
