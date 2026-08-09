# AI-ML

A collection of self-contained machine learning notebooks and mini-projects, each covering a classic algorithm or workflow (EDA, regression, classification, ensembles) using `scikit-learn` and standard data-science tooling.

## Projects

| Folder | Topic | Summary |
|---|---|---|
| [EDA](EDA) | Exploratory Data Analysis | EDA and preprocessing on the Algerian Forest Fires dataset; includes a cleaning pass and a model-training notebook regressing `area` (burned area). |
| [Regression-EDA-Feature-Engg](Regression-EDA-Feature-Engg) | EDA / Feature Engineering | Cleans and type-casts the raw Algerian Forest Fires dataset (region splitting, dtype fixes, binary `Classes` encoding), producing a cleaned CSV used for downstream modeling. |
| [simple linear regression](simple%20linear%20regression) | Simple Linear Regression | Height-vs-Weight regression with scaling, OLS diagnostics via `statsmodels`, and evaluation (has a known bug: test features aren't scaled before prediction). |
| [simple-lr-project-2](simple-lr-project-2) | Simple Linear Regression | A second Height-vs-Weight walkthrough — feature scaling, training, residual analysis, and adjusted R². Test R² ≈ 0.78. |
| [Mult-Linear-Regression](Mult-Linear-Regression) | Multiple Linear Regression | Predicts a stock market index from interest rate and unemployment rate; highlights multicollinearity between the two predictors. |
| [Mult-LR-2](Mult-LR-2) | Multiple Linear Regression | Multiple linear regression on the California Housing dataset, with scaling, residual/assumption checks, and model persistence via `pickle` (test R² ≈ 0.89; note: the notebook accidentally predicts `Longitude` rather than `MedHouseVal`). |
| [Logistic-Regression](Logistic-Regression) | Logistic Regression | Binary and multiclass logistic regression on synthetic data, with hyperparameter tuning via `GridSearchCV` and `RandomizedSearchCV`. |
| [KNN](KNN) | K-Nearest Neighbors | `KNeighborsClassifier` (accuracy 0.907) and `KNeighborsRegressor` (R² 0.919) on synthetic datasets. |
| [Naive-bayes](Naive-bayes) | Naive Bayes | `GaussianNB` classifier on the Iris dataset — perfect (1.00) accuracy on the test set. |
| [Decision-tree](Decision-tree) | Decision Tree | `DecisionTreeClassifier` on the Iris dataset with tree visualization — perfect test-set classification. |
| [Diabetes-pred](Diabetes-pred) | Decision Tree Regression | `DecisionTreeRegressor` tuned with `GridSearchCV` to predict diabetes disease progression (test R² ≈ 0.32). |
| [support-vector-classifiers](support-vector-classifiers) | SVM Classification | Compares linear, RBF, and polynomial kernel SVCs on a synthetic binary classification dataset. |
| [SVR](SVR) | Support Vector Regression | SVR (RBF kernel) predicting `total_bill` on the seaborn `tips` dataset, tuned via `GridSearchCV` (test R² improves from 0.498 to 0.563). |
| [Random-forestclass](Random-forestclass) | Random Forest / Ensembles | Predicts travel package purchases; benchmarks Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting, then tunes a Random Forest via `RandomizedSearchCV` (test accuracy 0.928, ROC-AUC 0.829). |
| [Adaboost-model](Adaboost-model) | AdaBoost | Placeholder folder — notebook is currently empty. |

## General Requirements

Most projects rely on the standard Python data-science stack:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Some notebooks additionally use `statsmodels` or `plotly` — see each project's own `readme.md` for specifics.

## Usage

Each folder is self-contained: open its notebook(s) in Jupyter or VS Code and run the cells top to bottom. Datasets are either bundled as CSVs in the folder or loaded directly via `sklearn.datasets` / `seaborn.load_dataset`.
