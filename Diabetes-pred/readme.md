# Diabetes Progression Prediction

A regression model that predicts diabetes disease progression one year after baseline, using the [scikit-learn diabetes dataset](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html) and a `DecisionTreeRegressor` tuned via `GridSearchCV`.

## Dataset

Loaded via `sklearn.datasets.load_diabetes`. 442 patient samples, 10 baseline features (already mean-centered and scaled):

| Feature | Description |
|---|---|
| age | Age in years |
| sex | Sex |
| bmi | Body mass index |
| bp | Average blood pressure |
| s1 | tc, total serum cholesterol |
| s2 | ldl, low-density lipoproteins |
| s3 | hdl, high-density lipoproteins |
| s4 | tch, total cholesterol / HDL |
| s5 | ltg, possibly log of serum triglycerides level |
| s6 | glu, blood sugar level |

Target: a quantitative measure of disease progression one year after baseline.

## Approach

1. Load the dataset into a pandas `DataFrame` and split into train/test sets (`test_size=0.3`, `random_state=42`).
2. Explore feature correlations with a seaborn heatmap.
3. Fit a baseline `DecisionTreeRegressor`.
4. Tune hyperparameters with `GridSearchCV` (5-fold CV, `neg_mean_squared_error` scoring) over:
   - `criterion`: `squared_error`, `friedman_mse`, `absolute_error`, `poisson`
   - `splitter`: `best`, `random`
   - `max_depth`: `1, 2, 3, 4, 5, 10, 15, 20, 25`
   - `max_features`: `auto`, `sqrt`, `log2`
5. Evaluate the best estimator on the held-out test set and visualize the resulting tree with `plot_tree`.

## Results

Best hyperparameters found:

```
{'criterion': 'friedman_mse', 'max_depth': 2, 'max_features': 'log2', 'splitter': 'best'}
```

Test set performance:

| Metric | Value |
|---|---|
| Mean Squared Error | 3673.52 |
| R² Score | 0.32 |
| Mean Absolute Error | 49.80 |

## Requirements

- pandas
- scikit-learn
- seaborn
- matplotlib

## Usage

Run [code.ipynb](code.ipynb) end-to-end to reproduce the data loading, training, tuning, and evaluation steps.
