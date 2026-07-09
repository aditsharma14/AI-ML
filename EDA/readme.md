# EDA — Algerian Forest Fires (UPDATE)

Short description
- This folder contains the Algerian Forest Fires dataset (updated/cleaned) and two notebooks for exploration and model training.

## Files
- `Algerian_forest_fires_dataset_UPDATE.csv` — original dataset (raw).
- `Algerian_forest_fires_dataset_UPDATE_cleaned.csv` — cleaned and preprocessed version used in notebooks.
- `dataset_update.csv` — auxiliary/merged dataset created during preprocessing.
- `practical.ipynb` — exploratory data analysis (EDA), visualizations, and preprocessing steps.
- `model-training.ipynb` — model experiments and training (regression on `area`).

## Dataset (columns)
Typical columns include: `X`, `Y`, `month`, `day`, `FFMC`, `DMC`, `DC`, `ISI`, `temp`, `RH`, `wind`, `rain`, `area`.

Notes:
- The target used for regression is `area` (notebook applies a `log(area+1)` transform in some experiments).
- The cleaned CSV has missing-value handling, type conversions, and categorical encodings for `month`/`day`.

## Quick start
1. Create a Python environment (recommended):

	```bash
	python -m venv .venv
	.venv\Scripts\activate
	```

2. Install required packages:

	```bash
	pip install pandas numpy matplotlib seaborn scikit-learn jupyter
	```

3. Open the notebooks with Jupyter or VS Code and run cells:

	- `practical.ipynb` for EDA and preprocessing
	- `model-training.ipynb` for training and evaluation

## Reproducibility & next steps
- If you want, I can add a `requirements.txt` or a preprocessing script (`prepare_data.py`) to reproduce the cleaned CSV.
- I can also extract the model-training pipeline into a standalone script or notebook cells for easier reuse.

---

Maintainer: please report issues or request additions.

