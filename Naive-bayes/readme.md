# Naive Bayes Classifier — Iris Dataset

A simple classification example using scikit-learn's `GaussianNB` to classify iris flowers into their species based on sepal/petal measurements.

## Dataset

- **Source:** `sklearn.datasets.load_iris`
- **Features (X):** sepal length, sepal width, petal length, petal width
- **Target (y):** iris species — `0` (setosa), `1` (versicolor), `2` (virginica)
- **Samples:** 150, split 70/30 into train/test (`train_test_split`, `random_state=0`)

## Approach

1. Load the Iris dataset.
2. Split into training (105 samples) and test (45 samples) sets.
3. Fit a `GaussianNB` classifier on the training data.
4. Predict on the test set and evaluate with accuracy, confusion matrix, and classification report.

## Results

| Metric | Score |
|---|---|
| Accuracy | 1.00 |
| Precision / Recall / F1 (all classes) | 1.00 |

The model perfectly separates all three species on the held-out test set.

## Requirements

```
scikit-learn
seaborn
```

## Files

- `code.ipynb` — notebook with the full workflow above.
