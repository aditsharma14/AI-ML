# Support Vector Classifiers

This notebook demonstrates how to build and evaluate support vector classifiers (SVCs) using scikit-learn on a synthetic binary classification dataset.

## Overview

The notebook performs the following steps:

1. Creates a synthetic dataset using `make_classification`
2. Visualizes the data with Seaborn
3. Splits the data into training and test sets
4. Trains SVM models with different kernels
5. Evaluates the models using classification reports and confusion matrices

## Dataset

The dataset is generated using:

- `n_samples=1000`
- `n_features=2`
- `n_clusters_per_class=1`
- `n_redundant=0`

The data is stored in a pandas DataFrame and labeled as `label`.

## Models Used

The notebook compares three SVM kernels:

- Linear kernel: `SVC(kernel='linear')`
- RBF kernel: `SVC(kernel='rbf')`
- Polynomial kernel: `SVC(kernel='poly')`

## Evaluation

Each trained model is evaluated using:

- `classification_report`
- `confusion_matrix`

## Dependencies

Install the required packages with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

Open and run the notebook file:

- `code.ipynb`

You can run it in Jupyter Notebook or in VS Code with Python support.