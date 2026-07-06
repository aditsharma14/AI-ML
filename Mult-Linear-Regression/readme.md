# Multiple Linear Regression - Economic Index

Predicting a stock market index price from interest rate and unemployment rate using multiple linear regression.

## Dataset

`economic_index.csv` contains 24 monthly records (2016-2017) with the following columns:

| Column | Description |
|---|---|
| `year` | Year of observation |
| `month` | Month of observation |
| `interest_rate` | Interest rate (%) |
| `unemployment_rate` | Unemployment rate (%) |
| `index_price` | Stock market index price (target variable) |

## Workflow (`practical.ipynb`)

1. Load the dataset and drop unused columns (`year`, `month`, unnamed index column).
2. Explore the data with `describe()`, null checks, pairplots, scatter plots, and a correlation matrix.
3. Select `interest_rate` and `unemployment_rate` as independent features (`X`) and `index_price` as the target (`y`).
4. Visualize relationships between features and the target with `sns.regplot`.
5. Scale features with `StandardScaler` and split into train/test sets.
6. Fit a multiple linear regression model and evaluate its performance.

## Key Findings

- `interest_rate` is strongly positively correlated with `index_price` (0.94).
- `unemployment_rate` is strongly negatively correlated with `index_price` (-0.92).
- `interest_rate` and `unemployment_rate` are themselves strongly negatively correlated (-0.93), indicating multicollinearity between the two predictors.

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Usage

```bash
jupyter notebook practical.ipynb
```
