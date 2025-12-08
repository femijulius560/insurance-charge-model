# Healthcare Insurance Cost Analysis & Prediction
This project analyses healthcare insurance data to understand how personal attributes and geographic factors influence medical insurance charges. It also develops predictive model for estimating insurance costs for individuals and groups.


# Project Goals
1. Understand how personal attributes and geographic factors influence insurance costs.
2. Identify patterns and significant risk indicators through descriptive, correlation, and predictive analytics.
3. Build interpretable machine learning models capable of estimating insurance charges.


# Dataset Description
| Attribute | Description                        | Data Type   |
| --------- | ---------------------------------- | ----------- |
| age       | Policyholder's age                 | Integer     |
| sex       | Male/Female                        | Binary      |
| bmi       | Body Mass Index                    | Float       |
| children  | Number of dependents               | Integer     |
| smoker    | Smoking status                     | Binary      |
| region    | Geographic region in the U.S.      | Categorical |
| charges   | Insurance cost (target variable)   | Float       |


# ETL Pipeline
**Extract**
- Raw CSV dataset exported from Kaggle.
- Imported into Pandas for processing.

**Transform**
- No missing data
- One duplicate row removed
- 'charges' highly skewed
- A few extreme BMI values were retained as realistic to preserve natural variation; no other outlier removal performed.
- Log transformation of 'charges' to handle skewness.
- Creation of BMI_category (WHO classification): Underweight, Normal, Overweight, Obese.
- Categorical variables encoded

**Load**
- Cleaned dataset imported for analysis and modeling.


# Exploratory Data Analysis (EDA)
**Descriptive Statistics**
Average insurance charges by:
Age groups
Sex
Smoker status
Region
BMI_category

**Visualizations**
A series of exploratory and analytical visualizations were created to understand key patterns in the dataset:
- Distribution plots of major features
- Scatter plots (e.g., BMI vs. Charges)
- A four-panel static dashboard (dashboard_static.png) summarizing average charges across age groups, BMI categories, smoker status, and regional segments, included for GitHub compatibility

**Correlation Analysis**
A combined correlation report was produced, including heatmaps and feature correlation bars.

|   Type       |      Method Used        |
|------------  |------------------------ |
| Numerical    | Pearson correlation     |
| Binary       | Point-Biserial          |
| Categorical  | Correlation Ratio (η)   |




# Predictive Modeling
Models Evaluated:
- Linear Regression: Baseline linear model
- Random Forest Regressor: Ensemble-based model (selected as final)

# Methodology
- Train-test split applied.
- Categorical features encoded using one-hot encoding.
Model trained on log_charges to address skewness; predictions transformed back to actual charges with expm1().
- Target Transformation: Models trained on `log_charges` to handle skewness
- Inverse Transformation: Predictions converted back to actual charges using `expm1()`

# Evaluation Metrics
| Metric | Description                  |
|--------|----------------------------  |
| MAE    | Mean Absolute Error          |
| MSE    | Mean Squared Error           |
| RMSE   | Root Mean Squared Error      |
| R²     | Coefficient of Determination |

# Results
Random Forest outperformed Linear Regression across all metrics.

# Predictive Reporting
- Full Predictive Report (csv)
- Actual vs. predicted scatter plots
- Summary tables

**Feature Importance:**
- Random Forest identified smoker_yes as the most important predictor, followed by age, bmi, and other features.
- Visualized using a bar chart to clearly show contribution of top features.

**Key Insights:**
- **Smoking status** is the strongest predictor of high insurance costs.
- **Age** is a secondary contributor; insurance costs rise with older age groups.
- **BMI** affects charges, with overweight and obese individuals incurring higher predicted costs.
- **Regional differences** exist but are less impactful than lifestyle factors and age.


# Notes
The model was trained on log-transformed charges; all predictions are reported in actual dollar values.
Extreme BMI values were retained to maintain realistic variability in the dataset.
