# Travel Package Prediction — Random Forest Classifier

Predicts whether a customer will purchase a new travel package (`ProdTaken`) based on demographic, behavioral, and sales-interaction data, using a Random Forest Classifier (benchmarked against Logistic Regression, Decision Tree, and Gradient Boosting).

## Dataset

`Travel.csv` — 4,888 rows, 20 columns, from a tourism company's customer records.

| Column | Description |
|---|---|
| CustomerID | Unique customer identifier (dropped before modeling) |
| ProdTaken | Target — 1 if customer purchased the package, 0 otherwise |
| Age | Customer age |
| TypeofContact | How the customer was contacted (Self Enquiry / Company Invited) |
| CityTier | Tier of the customer's city (1, 2, 3) |
| DurationOfPitch | Duration (minutes) of the sales pitch |
| Occupation | Customer occupation |
| Gender | Customer gender |
| NumberOfPersonVisiting | Number of people accompanying the customer |
| NumberOfFollowups | Number of follow-ups by the salesperson |
| ProductPitched | Type of product pitched |
| PreferredPropertyStar | Preferred hotel star rating |
| MaritalStatus | Marital status |
| NumberOfTrips | Average annual number of trips |
| Passport | Whether the customer holds a passport (0/1) |
| PitchSatisfactionScore | Satisfaction score for the sales pitch |
| OwnCar | Whether the customer owns a car (0/1) |
| NumberOfChildrenVisiting | Number of children accompanying the customer |
| Designation | Customer's job designation |
| MonthlyIncome | Customer's monthly income |

Class distribution is imbalanced: 3,968 negatives vs. 920 positives (~18.8% positive class).

## Workflow (`code.ipynb`)

1. **EDA** — inspect shape, dtypes, summary statistics, and missing values.
2. **Data cleaning**
   - Fixed inconsistent category labels: `Fe Male` → `Female`, `Single` → `Unmarried`.
   - Imputed missing values in `NumberOfFollowups`, `PreferredPropertyStar`, `NumberOfTrips`, `NumberOfChildrenVisiting`, and `MonthlyIncome` using the column mode.
   - Dropped `CustomerID` (non-predictive identifier).
3. **Feature engineering** — combined `NumberOfChildrenVisiting`, `NumberOfFollowups`, and `NumberOfTrips` into a single `TotalVisiting` feature, replacing the three original columns.
4. **Feature split** — 16 predictors (6 categorical, remainder numerical) vs. target `ProdTaken`.
5. **Preprocessing** — `train_test_split` (80/20, `random_state=42`), followed by a `ColumnTransformer` applying `OneHotEncoder(drop='first')` to categorical features and `StandardScaler` to numerical features.
6. **Baseline modeling** — trained and compared Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting using Accuracy, F1, Precision, Recall, and ROC-AUC on train/test sets.
7. **Hyperparameter tuning** — `RandomizedSearchCV` (100 iterations, 3-fold CV) over `n_estimators`, `max_depth`, `max_features`, and `min_samples_split` for the Random Forest.
8. **Final model** — `RandomForestClassifier(n_estimators=1000, min_samples_split=2, max_features=7, max_depth=None)`.
9. **Evaluation** — ROC curve plotted for the tuned model and saved as `auc.png`.

## Results (tuned Random Forest)

| Metric | Train | Test |
|---|---|---|
| Accuracy | 1.0000 | 0.9284 |
| F1 score | 1.0000 | 0.9233 |
| Precision | 1.0000 | 0.9549 |
| Recall | 1.0000 | 0.6649 |
| ROC-AUC | 1.0000 | 0.8286 |

The perfect training scores alongside a visible train/test gap indicate the model is overfitting; test recall (0.66) is notably lower than test precision (0.95), meaning the model misses a meaningful share of actual purchasers despite being highly accurate overall — likely a consequence of class imbalance (~19% positive rate) and unconstrained tree depth.

## Files

- `code.ipynb` — full notebook (EDA, preprocessing, modeling, tuning, evaluation)
- `Travel.csv` — raw dataset
- `auc.png` — ROC curve of the final tuned Random Forest model

## Requirements

`pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `scikit-learn`
