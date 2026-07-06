# Simple Linear Regression — Height vs Weight

A basic simple linear regression walkthrough using the classic Height/Weight dataset, built with `pandas`, `matplotlib`/`seaborn`, and `scikit-learn`.

## Files

- `height-weight.csv` — 23 rows of `Weight` (kg) and `Height` (cm) samples.
- `practical.ipynb` — end-to-end notebook: EDA, train/test split, feature scaling, model fitting, evaluation, and OLS diagnostics.

## Workflow

1. **Load & explore** — `df.head()`, `df.describe()`, scatter plots of Height vs Weight, and a correlation heatmap (`df.corr()` ≈ 0.93 between Weight and Height).
2. **Prepare data** — `X = Weight` (feature), `Y = Height` (target). Split with `train_test_split(test_size=0.25, random_state=42)`.
3. **Scale features** — `StandardScaler` fit/transform on `X_train`.
4. **Fit model** — `sklearn.linear_model.LinearRegression` on the scaled training data.
   - Coefficient (slope): `285.54`
   - Intercept: `1410.08`
5. **Visualize fit** — scatter of training data with the regression line overlaid.
6. **Evaluate** — `mean_squared_error`, `mean_absolute_error`, `root_mean_squared_error`, and `r2_score` on the test set.
7. **Cross-check with statsmodels** — `sm.OLS(Y_train, X_train).fit()` for a detailed regression summary (coefficients, p-values, confidence intervals).
8. **Predict new data** — example prediction for an unseen `Weight` value using the fitted scaler + model.

## Known issue

`X_test` is reshaped but **not** passed through the fitted `scaler` before calling `regression.predict(X_test)`, while the model was trained on scaled features. This mismatch is why the test-set metrics (MSE/MAE/RMSE and a large negative R²) look off — `X_test` should be transformed with `scaler.transform(X_test)` before prediction to get valid results.

## Requirements

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`
