# Simple Linear Regression — Height vs Weight

A simple linear regression project that predicts **Height** from **Weight** using scikit-learn, built on the `height-weight.csv` dataset.

## Dataset

`height-weight.csv` contains 23 records with two columns:

| Column | Description |
|--------|-------------|
| `Weight` | Independent feature (X) |
| `Height` | Dependent/target feature (y) |

## Workflow

### 1. Imports & Setup
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

### 2. Load Data
```python
dataset = pd.read_csv('height-weight.csv')
```

### 3. Exploratory Visualization
A scatter plot of `Height` vs `Weight` is drawn with `sns.scatterplot` to check for a linear relationship.

### 4. Feature Selection
```python
X = dataset['Weight']  # independent feature
y = dataset['Height']  # dependent feature
```

### 5. Train/Test Split
```python
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
```
- `X.shape` → `(23,)`
- `train_X.shape, test_X.shape, train_y.shape, test_y.shape` → `((18,), (5,), (18,), (5,))`

### 6. Feature Scaling
`StandardScaler` is fit on the training data and applied to both train and test sets:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X.values.reshape(-1, 1))
test_X = scaler.transform(test_X.values.reshape(-1, 1))
```

### 7. Visualize Training Data
A scatter plot of scaled `train_X` vs `train_y` is plotted (in red, then pink) to confirm the relationship persists after scaling.

### 8. Train the Model
```python
from sklearn.linear_model import LinearRegression
LinearRegressionModel = LinearRegression()
LinearRegressionModel.fit(train_X, train_y)
```

**Learned parameters:**
- Coefficient (slope): `17.03440872`
- Intercept: `157.5`

### 9. Plot the Regression Line
The fitted line is plotted over the training scatter points using `LinearRegressionModel.predict(train_X)`.

### 10. Predict on Test Data
```python
y_pred_test = LinearRegressionModel.predict(test_X)
```

| Predicted Height | Actual Height |
|---|---|
| 161.08 | 177 |
| 161.08 | 170 |
| 129.30 | 120 |
| 177.46 | 182 |
| 148.57 | 159 |

The test predictions vs actuals are also visualized with a scatter + line plot.

### 11. Performance Metrics
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
```

| Metric | Value |
|--------|-------|
| MSE | 109.7759 |
| MAE | 9.8227 |
| RMSE | 10.4774 |
| R² Score | 0.7770 |
| Adjusted R² | 0.7026 |

**Adjusted R² calculation:**
```python
n = len(test_y)              # number of samples
p = test_X.shape[1]          # number of features
r2 = score

adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
```

### 12. Prediction on New Data
Predicting height for a new weight value of **80**:
```python
scaled_weight = scaler.transform([[80]])       # -> [[0.32350772]]
LinearRegressionModel.predict([scaled_weight[0]])   # -> 163.01076266
```

### 13. Residual Analysis
```python
residuals = test_y - y_pred_test
```

| Index | Residual |
|-------|----------|
| 15 | 15.915329 |
| 9 | 8.915329 |
| 0 | -9.304156 |
| 8 | 4.543549 |
| 17 | 10.434926 |

The residual distribution is plotted with a KDE plot:
```python
sns.displot(residuals, kind='kde')
```

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Run

1. Ensure `height-weight.csv` is in the same directory.
2. Open `practical.ipynb` in Jupyter.
3. Run all cells sequentially.

## Summary

A single-feature (`Weight`) linear regression model was trained to predict `Height`. The model achieves an R² of ~0.78 on the test set, explaining a good portion of the variance, with residuals showing a roughly symmetric spread around zero.
