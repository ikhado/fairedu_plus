# FaireduPlus

Enhancing Intersectional Fairness in Education-Focused Machine Learning Using Synthetic Data
## Requirements
- Python 3.9+
- Packages: pandas, numpy, scikit-learn, aif360, ctgan, sdgx, statsmodels
- For LLM generation set `OPENAI_API_KEY` (used by `generate_sdg.generate_by_llm`).

## Data layout
- `dataset_folder`: https://zenodo.org/records/17933909. Test data is expected at `<dataset_folder>/<dataset_name>/test_<dataset_name>.csv`.
- Generated synthetic files are merged into the file name passed via `--merged-output-file-name`.

## Quick start
```bash
python fairedu_plus.py \
  --dataset-name student_dropout \
  --dataset-folder /home/ad/m4do/proj/fairedu_plus/original_dataset \
  --generator LLM \
  --merged-output-file-name merged_output.csv \
  --seed 42
```

## CLI options (`fairedu_plus.py`)
- `--dataset-name` (`student_dropout`, `student_oulad`, `student_performance`, `DNU`)
- `--dataset-folder` path to dataset root used to locate the test CSV
- `--generator` choose `LLM` or `CTGAN`
- `--merged-output-file-name` name for merged synthetic output
- `--run-splitted-file` / `--no-run-splitted-file` choose split vs combined training files
- `--seed` random seed for reproducibility

## Additional examples
- CTGAN without split files:
```bash
python fairedu_plus.py --dataset-name student_dropout --generator CTGAN --no-run-splitted-file
```

- Using OULAD dataset with explicit dataset folder:
```bash
python fairedu_plus.py --dataset-name student_oulad --dataset-folder ./dataset
```

## Citation
If you use this work, please cite the SSRN preprint:
```
SSRN Scholarly Paper 5290738. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5290738
```

## Acknowledgements

This work builds upon the following open-source projects:
- https://github.com/hitsz-ids/synthetic-data-generator
- https://github.com/fairnesstest/LTDD
