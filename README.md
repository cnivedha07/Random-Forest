# Random Forest Customer Churn Prediction

A machine learning classification project that uses **Random Forest** to predict whether a telecom customer is likely to churn based on customer attributes and service usage information.

The project covers the complete workflow from **data cleaning and exploratory data analysis (EDA) to feature engineering, model training, evaluation, and customer-level prediction**.

---

## Project Overview

Customer churn prediction is a common machine learning problem in the telecom industry. Identifying customers who are likely to leave can help companies understand churn patterns and take appropriate retention actions.

In this project, a **Random Forest Classifier** is trained on the Telco Customer Churn dataset to predict whether a customer will:

* `0` → No Churn
* `1` → Churn

The model uses:

* `MonthlyCharges`
* `tenure`
* `Contract`

The original assignment also mentioned `SupportCalls`, but the dataset used in this project does **not** contain that feature. Therefore, it was not artificially replaced with another feature.

---

## Objectives

The main objectives of this project are to:

1. Understand and clean the telecom customer dataset.
2. Perform exploratory data analysis.
3. Analyze the relationship between customer characteristics and churn.
4. Encode categorical features for machine learning.
5. Train a Random Forest classification model.
6. Evaluate the model using multiple classification metrics.
7. Analyze feature importance.
8. Use the trained model to predict churn for a new customer.

---

## Dataset

The project uses the **Telco Customer Churn** dataset.

The dataset contains information about telecom customers, including variables related to:

* Customer tenure
* Monthly charges
* Total charges
* Contract type
* Services
* Demographic information
* Churn status

### Target Variable

`Churn`

| Original Value | Encoded Value |
| -------------- | ------------: |
| No             |             0 |
| Yes            |             1 |

---

## Technologies Used

* **Python**
* **Pandas** – data manipulation and preprocessing
* **NumPy** – numerical operations
* **Matplotlib** – data visualization
* **Seaborn** – statistical visualization
* **Scikit-learn** – machine learning and evaluation
* **Google Colab / Jupyter Notebook**

---

## Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Statistical Analysis
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Random Forest Training
   ↓
Predictions
   ↓
Model Evaluation
   ↓
Feature Importance
   ↓
New Customer Prediction
```

---

## 1. Data Loading and Understanding

The dataset is loaded using Pandas.

Initial inspection includes:

* Dataset shape
* Column names
* Data types
* Missing values
* Duplicate records
* Descriptive statistics
* Target distribution
* Contract distribution

One important data-quality issue was identified:

`TotalCharges` was stored as an `object` instead of a numerical column.

It was converted using:

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

Rows containing invalid/missing `TotalCharges` values were then removed.

---

## 2. Exploratory Data Analysis

Several visualizations were created to understand churn behavior.

### Churn Distribution

The dataset is imbalanced:

* Approximately **73.4%** of customers did not churn.
* Approximately **26.6%** of customers churned.

This is important because a model can obtain reasonably high accuracy simply by favoring the majority class.

Therefore, accuracy alone is not sufficient for evaluating the model.

---

### Monthly Charges vs Churn

Customers who churned had higher monthly charges in the dataset.

The median monthly charge was approximately:

* No Churn: **64.45**
* Churn: **79.65**

---

### Tenure vs Churn

Customers who churned generally had shorter tenure.

The median tenure was approximately:

* No Churn: **38 months**
* Churn: **10 months**

This indicates that newer customers represented an important churn group in this dataset.

---

### Contract Type vs Churn

Contract type showed a substantial difference in churn rates.

Approximate churn rates observed in the notebook:

| Contract       | Churn Rate |
| -------------- | ---------: |
| Month-to-month |      42.7% |
| One year       |      11.3% |
| Two year       |       2.8% |

This shows a strong relationship between contract type and observed churn in the dataset.

---

### Correlation Analysis

The correlation heatmap showed:

* `tenure` had a negative correlation with churn of approximately **-0.35**
* `MonthlyCharges` had a positive correlation with churn of approximately **0.19**
* `TotalCharges` had a negative correlation with churn of approximately **-0.20**
* `tenure` and `TotalCharges` were strongly correlated at approximately **0.83**

Correlation was treated only as a descriptive analysis because correlation does not capture all types of relationships.

---

## 3. Feature Selection

The assignment specified four features:

```text
MonthlyCharges
tenure
SupportCalls
Contract
```

However, the dataset did not contain a `SupportCalls` column.

Instead of incorrectly substituting another feature, the model was trained using the three features actually available and relevant to the assignment:

```python
features = [
    "MonthlyCharges",
    "tenure",
    "Contract"
]
```

---

## 4. Feature Encoding

`Contract` is a categorical feature containing:

* Month-to-month
* One year
* Two year

One-hot encoding was applied:

```python
X = pd.get_dummies(
    X,
    columns=["Contract"],
    drop_first=True
)
```

This produces numerical features suitable for the Random Forest model.

The dropped category is `Month-to-month`.

---

## 5. Train-Test Split

The dataset was divided into:

* **80% training data**
* **20% testing data**

Stratification was used to preserve the churn/non-churn class distribution:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

---

## 6. Random Forest Model

A Random Forest Classifier with **100 decision trees** was used.

```python
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
```

Random Forest combines predictions from multiple decision trees rather than relying on a single tree.

This helps reduce the variance associated with individual decision trees.

---

## 7. Model Evaluation

The model was evaluated using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score
* Classification Report

### Accuracy

The model achieved approximately:

**75% test accuracy**

However, this number should be interpreted carefully.

Because approximately 73% of the dataset belongs to the No Churn class, a classifier that always predicted No Churn would already achieve around 73% accuracy.

Therefore, the approximately 75% accuracy indicates only a modest improvement over the majority-class baseline.

---

### Confusion Matrix

The test-set confusion matrix was:

|                     | Predicted No Churn | Predicted Churn |
| ------------------- | -----------------: | --------------: |
| **Actual No Churn** |                883 |             150 |
| **Actual Churn**    |                202 |             172 |

Where:

* **True Negative:** 883
* **False Positive:** 150
* **False Negative:** 202
* **True Positive:** 172

The model therefore missed a substantial number of customers who actually churned.

---

### Churn-Class Performance

Approximate performance for the churn class:

| Metric    | Value |
| --------- | ----: |
| Precision |  0.53 |
| Recall    |  0.46 |
| F1-score  |  0.49 |

The recall of approximately **0.46** means the model identified fewer than half of the customers who actually churned in the test set.

This is an important limitation of the current model.

---

## 8. Feature Importance

Random Forest provides feature importance values based on impurity reduction across the trees.

The most important features in this model were approximately:

| Feature           |           Importance |
| ----------------- | -------------------: |
| MonthlyCharges    |                 0.62 |
| tenure            |                 0.29 |
| Contract features | Remaining importance |

`MonthlyCharges` and `tenure` were the two most influential features according to the model's impurity-based feature importance.

Feature importance should be interpreted as model-specific importance rather than causal importance.

---

## 9. New Customer Prediction

The trained model was also tested on a new customer with:

```text
MonthlyCharges = 70
Tenure = 8 months
Contract = Month-to-month
```

The assignment also specified `SupportCalls = 4`, but this feature was unavailable in the dataset and therefore was not used.

The model predicted:

```text
Predicted Class: No Churn
No Churn Probability: ~0.75
Churn Probability: ~0.25
```

The customer has characteristics associated with higher observed churn in the EDA, particularly short tenure and a month-to-month contract. However, the model's predicted churn probability remained below 0.5.

---

## 10. Key Findings

The analysis identified several patterns in the dataset:

* Churned customers generally had **higher monthly charges**.
* Churned customers generally had **shorter tenure**.
* Month-to-month customers had substantially higher observed churn rates than customers on longer contracts.
* The dataset is **class imbalanced**.
* Random Forest achieved approximately **75% accuracy**, but accuracy alone does not adequately represent churn-class performance.
* The model's churn recall was approximately **46%**, meaning many actual churners were missed.
* `MonthlyCharges` and `tenure` were the most important features according to the model's feature importance.

---

## Limitations

This project has several limitations.

### 1. Limited Feature Set

Only three features were used:

```text
MonthlyCharges
tenure
Contract
```

The dataset did not contain `SupportCalls`, so that assignment feature could not be included.

### 2. Class Imbalance

Approximately 73% of customers did not churn.

This makes accuracy potentially misleading.

### 3. Moderate Churn Detection

The model achieved approximately 46% recall for the churn class.

Therefore, it misses a considerable number of customers who actually churn.

### 4. Limited Model Tuning

The Random Forest was trained with:

```python
n_estimators = 100
```

No extensive hyperparameter optimization was performed.

### 5. Feature Importance Is Not Causality

A high Random Forest feature-importance value does not mean that the feature causes churn.

---

## Possible Improvements

Future versions of the project could improve the model by:

* Including additional relevant customer features.
* Handling class imbalance using class weights or resampling techniques.
* Performing hyperparameter tuning.
* Using cross-validation.
* Comparing Random Forest with other classifiers.
* Evaluating ROC-AUC and Precision-Recall AUC.
* Optimizing the classification threshold for churn detection.
* Performing more detailed feature engineering.
* Using permutation importance or SHAP for model interpretation.

---

## Project Structure

```text
Random-Forest-Customer-Churn/
│
├── Random_Forest_Customer_Churn.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
```

> If the dataset is not included in your GitHub repository, remove the CSV from this structure and mention the dataset source separately.

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Random-Forest-Customer-Churn
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open

```text
Random_Forest_Customer_Churn.ipynb
```

### 5. Make sure the dataset is available

The notebook expects:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

in the appropriate working directory.

---

## Conclusion

This project demonstrates an end-to-end **Random Forest classification workflow for customer churn prediction**.

The analysis shows that `MonthlyCharges`, `tenure`, and `Contract` contain useful information about churn. However, the current model provides only moderate churn-detection performance
