# Mult-LR-2: Multiple Linear Regression on California Housing Data

## Overview
This project demonstrates multiple linear regression using the **California Housing dataset** from `sklearn.datasets`. It covers the full workflow: loading data, exploratory data analysis (EDA), train/test split, feature scaling, model training, evaluation, and model persistence.

## Dataset
The dataset is loaded via `sklearn.datasets.fetch_california_housing()` and contains 20,640 rows with 8 numeric features (no missing values), derived from the 1990 U.S. census.

| Feature | Description |
|---|---|
| MedInc | Median income in block group |
| HouseAge | Median house age in block group |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | Block group population |
| AveOccup | Average number of household members |
| Latitude | Block group latitude |
| Longitude | Block group longitude |

The dataset's official target (`target_names`) is `MedHouseVal` (median house value, in $100,000s). However, in this notebook the DataFrame `df` is built only from the 8 feature columns (the `MedHouseVal` target array is never attached to `df`). As a result:

- `X = df.iloc[:, :-1]` → `MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude` (7 features)
- `y = df.iloc[:, -1]` → `Longitude`

So the model actually trained in this notebook predicts **Longitude** from the other 7 features, not the median house value. Keep this in mind when interpreting the results below.

## Workflow (`practice.ipynb`)
1. **Load data** — `fetch_california_housing()`, inspect keys, `DESCR`, `feature_names`, `target_names`.
2. **Build DataFrame** — `pd.DataFrame(california.data, columns=california.feature_names)`.
3. **EDA**
   - `df.describe()`, `df.info()` — summary statistics, no nulls.
   - `df.corr()` + `sns.heatmap()` — correlation matrix (notably `AveRooms`/`AveBedrms` correlate at 0.85, and `Latitude`/`Longitude` at -0.92).
4. **Feature/target split** — `X` = first 7 columns, `y` = `Longitude` (last column).
5. **Train/test split** — `train_test_split(X, y, test_size=0.33, random_state=10)`
   - Train: 13,828 rows, Test: 6,812 rows.
6. **Feature scaling** — `StandardScaler`, fit on train, applied to both train and test.
7. **Model training** — `sklearn.linear_model.LinearRegression`.
8. **Prediction** — on scaled test data.
9. **Evaluation metrics**:
   - MAE: `0.4943`
   - MSE: `0.4562`
   - RMSE: `0.6754`
   - R² score: `0.8857`
10. **Assumption checks (residual analysis)**:
    - Scatter plot of `y_test` vs `y_pred` (linearity check).
    - Residuals (`y_test - y_pred`) computed.
    - KDE plot of residuals (checking for normal distribution).
    - Scatter plot of predictions vs residuals (checking for homoscedasticity/uniform spread).
11. **Model persistence** — model saved with `pickle` to `regressor.pkl`.

## Model Coefficients
Trained on standardized features, in order `[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude]`:

```
coef_ = [-0.3833, -0.2020, 0.4692, -0.2691, -0.0515, 0.0253, -1.9252]
intercept_ = -119.5711
```

## Files
| File | Description |
|---|---|
| `practice.ipynb` | Main notebook containing the full regression workflow described above. |
| `regressor.pkl` | Pickled `LinearRegression` model saved from this notebook (via `pickle.dump`). |
| `model.pkl` | Present in the folder but empty/corrupted — loading it (`pickle.load(open('model.pkl','rb'))`) raises `EOFError: Ran out of input`. |
| `readme.md` | This file. |

## Requirements
- Python 3
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

## Notes / Possible Fixes
- To actually predict `MedHouseVal` as the dataset intends, add the target column to `df` (e.g. `df['MedHouseVal'] = california.target`) and select `X`/`y` accordingly, rather than relying on `Longitude` as the last column.
- `model.pkl` appears to be an empty or corrupted file and should be regenerated (e.g., re-run the `pickle.dump` step and save to `model.pkl`) if it's meant to be used elsewhere.
