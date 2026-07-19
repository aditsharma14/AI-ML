# Support Vector Regression (SVR) — Tips Dataset

This project uses Support Vector Regression to predict the `total_bill` from the seaborn `tips` dataset.

## Dataset

The [seaborn `tips` dataset](https://github.com/mwaskom/seaborn-data), containing 244 rows of restaurant billing records:

| Column       | Description                              |
|--------------|-------------------------------------------|
| total_bill   | Total bill amount (target variable)       |
| tip          | Tip amount                                |
| sex          | Gender of the bill payer                  |
| smoker       | Whether the party included a smoker       |
| day          | Day of the week                           |
| time         | Lunch or Dinner                           |
| size         | Size of the dining party                  |

```python
import seaborn as sns
df = sns.load_dataset('tips')
```

## Workflow

### 1. Exploratory Data Analysis
- Inspected the dataset with `df.info()` and `df.describe()`.
- Checked category balance with `value_counts()` on `sex` and `smoker`.

### 2. Feature/Target Split
```python
X = df[['tip', 'sex', 'smoker', 'day', 'time', 'size']]
y = df['total_bill']
```

### 3. Train/Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

### 4. Feature Encoding
- **Label Encoding** for binary categorical columns (`sex`, `smoker`, `time`) using separate `LabelEncoder` instances, fit on train and applied to test.
- **One-Hot Encoding** for the multi-class `day` column via `ColumnTransformer` (with `drop='first'` to avoid the dummy variable trap), leaving the remaining columns untouched (`remainder='passthrough'`).

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[('onehot', OneHotEncoder(drop='first'), [3])],
    remainder='passthrough'
)
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)
```

### 5. Baseline Model
Trained a default `SVR` (RBF kernel) model:

```python
from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
```

**Baseline results:**
| Metric | Value |
|--------|-------|
| R² Score | 0.498 |
| MAE | 4.463 |

### 6. Hyperparameter Tuning (GridSearchCV)
Searched over `C`, `gamma`, and `kernel` to improve performance:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)
```

**Best parameters:** `{'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}`

**Tuned results:**
| Metric | Value |
|--------|-------|
| R² Score | 0.563 |
| MAE | 4.260 |

## Results Summary

| Model              | R² Score | MAE   |
|--------------------|----------|-------|
| SVR (default)      | 0.498    | 4.463 |
| SVR (tuned via GridSearchCV) | 0.563 | 4.260 |

Hyperparameter tuning improved both R² and MAE over the default SVR model.

## Requirements
- pandas
- numpy
- seaborn
- scikit-learn

## Files
- `code.ipynb` — full notebook with EDA, preprocessing, model training, and evaluation.
