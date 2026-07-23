# Data Cleaning & Preprocessing for Machine Learning (Titanic Dataset)

## 📌 Project Overview
This project demonstrates essential data preprocessing techniques on the Titanic dataset using Python, Pandas, NumPy, and Scikit-Learn.

### Steps Performed:
1. **Exploratory Data Analysis:** Checked missing values, data types, and summary statistics.
2. **Handling Missing Data:** Imputed missing `Age` values using median and `Embarked` using mode. Dropped the `Cabin` column due to high missingness.
3. **Categorical Encoding:** Mapped `Sex` to binary values and applied One-Hot Encoding (`pd.get_dummies`) to `Embarked`.
4. **Outlier Removal:** Identified and removed extreme outliers in `Fare` using the Interquartile Range (IQR) method.
5. **Feature Scaling:** Applied `StandardScaler` to normalize numerical features (`Age` and `Fare`).

---

## 🎯 Interview Questions & Answers

### 1. What are the different types of missing data?
* **MCAR (Missing Completely at Random):** Missingness is completely independent of any data.
* **MAR (Missing at Random):** Missingness depends on observed data, not the missing data itself.
* **MNAR (Missing Not at Random):** Missingness depends directly on the value of the missing data.

### 2. How do you handle categorical variables?
* **Ordinal Encoding:** Used when categories have a natural ranking (e.g., Low, Medium, High).
* **One-Hot Encoding:** Creates binary columns for nominal variables without an inherent order.

### 3. What is the difference between normalization and standardization?
* **Normalization (Min-Max):** Rescales data to a range of [0, 1]. Best when data does not follow a normal distribution.
* **Standardization (Z-Score):** Centers data around a mean of 0 with standard deviation of 1. Best when features follow a Gaussian distribution or for distance-based algorithms.

### 4. How do you detect outliers?
* **Visualizations:** Box plots, scatter plots, and histograms.
* **Statistical Methods:** IQR method (values outside $Q1 - 1.5 \times IQR$ or $Q3 + 1.5 \times IQR$) and Z-score method ($|Z| > 3$).

### 5. Why is preprocessing important in ML?
Preprocessing removes noise, scales variables fairly, and formats data so machine learning algorithms can converge faster and yield accurate predictions.

### 6. What is One-Hot Encoding vs. Label Encoding?
* **Label Encoding:** Converts categories into integers (0, 1, 2). Can cause models to assume an artificial order.
* **One-Hot Encoding:** Creates separate binary columns for each category, avoiding implicit ranking.

### 7. How do you handle data imbalance?
* **Resampling:** Oversampling the minority class (e.g., SMOTE) or undersampling the majority class.
* **Algorithmic:** Adjusting class weights or using evaluation metrics like F1-score/ROC-AUC instead of accuracy.

### 8. Can preprocessing affect model accuracy?
Yes, significantly. Proper scaling and encoding prevent biased features from dominating the model, preventing data leakage and drastically boosting performance.
