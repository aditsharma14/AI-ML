# K-Nearest Neighbors (KNN)

Implementations of KNN for both classification and regression using scikit-learn, with synthetic datasets.

## Files

- `classifier.ipynb` — KNN Classifier
- `regression.ipynb` — KNN Regressor

## classifier.ipynb

Binary classification on a synthetic dataset generated with `make_classification`:
- 1000 samples, 3 features (1 redundant), 2 classes, `random_state=999`
- Train/test split: 70/30 (`random_state=42`)
- Model: `KNeighborsClassifier(n_neighbors=5, algorithm='auto')`

**Results:**
- Accuracy: **0.907**
- Confusion matrix:
  ```
  [[144  18]
   [ 10 128]]
  ```
- Precision/Recall/F1 (macro avg): 0.91 / 0.91 / 0.91

## regression.ipynb

Regression on a synthetic dataset generated with `make_regression`:
- 1000 samples, 2 features, noise=10, `random_state=42`
- Train/test split: 67/33 (`random_state=42`)
- Model: `KNeighborsRegressor(n_neighbors=6, algorithm='auto')`

**Results:**
- R² score: **0.919**
- Mean Absolute Error: **9.01**
- Mean Squared Error: **127.46**

## Key Takeaways

- KNN is a non-parametric, instance-based algorithm — predictions are made by majority vote (classification) or averaging (regression) over the `n_neighbors` nearest points.
- `algorithm='auto'` lets scikit-learn choose the most efficient neighbor search method (ball tree, KD tree, or brute force) based on the data.
- Both notebooks use synthetic data purely for demonstrating the KNN workflow: data generation → train/test split → fit → predict → evaluate.
