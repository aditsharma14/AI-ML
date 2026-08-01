# Decision Tree Classifier — Iris Dataset

A simple example of training a `DecisionTreeClassifier` on the classic Iris dataset using scikit-learn.

## What it does

1. Loads the Iris dataset (`sklearn.datasets.load_iris`) — 150 samples, 4 features (sepal length/width, petal length/width), 3 classes (setosa, versicolor, virginica).
2. Builds a features DataFrame `X` and target array `y`.
3. Splits the data into train/test sets (80/20, `random_state=42`).
4. Trains a `DecisionTreeClassifier` on the training data.
5. Visualizes the fitted tree with `sklearn.tree.plot_tree`.
6. Predicts on the test set and evaluates performance with a confusion matrix and classification report.

## Results

On the held-out test set (30 samples), the model achieves perfect classification:

- Confusion matrix: all predictions correct, no misclassifications.
- Precision, recall, and F1-score: 1.00 for all three classes.

## Requirements

- pandas
- numpy
- matplotlib
- scikit-learn

## Usage

Open and run `code.ipynb` top to bottom in Jupyter.
