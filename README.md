
# HR Analytics: Predicting Candidate Joining Behavior

This repository contains an end-to-end data science workflow designed to predict whether a candidate will accept a job offer and successfully join the organization (`Joined`) or back out (`Not Joined`). By leveraging historical recruitment data and **Logistic Regression**, this project helps Human Resources (HR) departments proactively identify high-risk candidates, optimize recruitment pipelines, and minimize business disruption.

---

## 📌 Project Overview
In modern talent acquisition, candidates declining offers at the last minute or "ghosting" after acceptance results in significant losses in time, productivity, and recruitment costs. This project addresses this challenge through a statistical approach that:
1. **Explores Key Drivers**: Analyzes how notice periods, joining bonuses, compensation hikes (`Percent difference CTC`), and business units (`LOB`) impact candidate behavior.
2. **Addresses Multicollinearity**: Uses a robust feature-pruning mechanism via **Variance Inflation Factor (VIF)**.
3. **Builds a Predictive Model**: Develops an interpretable **Logistic Regression** model.
4. **Optimizes for Business Cost**: Introduces a cost-minimization framework to select the optimal classification threshold based on the real financial impacts of False Positives and False Negatives.

---

## Dataset & Features
The model utilizes the following features from the candidate profile (`hr_data.csv`):

- **Target Variable**: 
  - `Status_binary`: `0` for **Joined**, `1` for **Not Joined** (The primary focus of prediction).

- **Predictor Variables**:
  - `DOJ Extended`: Whether the Date of Joining was extended (`Yes`/`No`).
  - `Notice period`: The official notice period required by the candidate's previous employer.
  - `Candidate Source`: Sourcing channel (e.g., Agency, Employee Referral, Direct).
  - `Percent difference CTC`: Percentage increase in Cost-to-Company (CTC) offered compared to the current salary.
  - `Age`: Age of the candidate.
  - `Offered band`: The hierarchy level/grade of the job offer.
  - `LOB`: Line of Business / Department.
  - `Joining Bonus`: Whether a sign-on bonus was provided.

---

##  Core Technical Pipeline

### 1. Exploratory Data Analysis (EDA)
- Computes distribution percentages and visualizes key relationships:
  - **Notice Period vs. HR Status**: Understanding if longer notice periods increase the likelihood of dropping out.
  - **Offered Band & LOB Trends**: Pinpointing specific departments or roles with higher attrition rates before joining.
  - **Joining Bonus Effectiveness**: Evaluating if financial incentives significantly improve joining rates.

### 2. Feature Engineering & Preprocessing
- One-hot encodes all categorical variables via `pd.get_dummies(..., drop_first=True)` to prevent the dummy variable trap.
- Evaluates interaction effects (e.g., combining `Age` and `Offered band`).

### 3. Multicollinearity Mitigation (VIF Filter)
- Includes a custom utility function `clean_X_for_logit()` that iteratively calculates and drops features with a **Variance Inflation Factor (VIF)** above a designated threshold (e.g., 10). This ensures stable coefficient estimates and reliable p-values for the logistic model.

### 4. Statistical Modeling
- Splits data into 80% training and 20% testing partitions using `train_test_split`.
- Fits a formal statistical model using `statsmodels.api.Logit` to gain access to summary tables containing coefficients, z-scores, and p-values for clear feature interpretability.

### 5. Advanced Evaluation & Cost-Based Optimization
- Renders standard diagnostic metrics:
  - **Confusion Matrix** (`draw_cm`)
  - **Receiver Operating Characteristic (ROC) Curve** & **AUC Score** (`draw_roc`)
- **Cost Minimization Framework**: Instead of blindly using a `0.5` probability threshold, the workflow iterates through candidate cutoffs (from `0.10` to `0.49`) to calculate the total business cost:
  $$\text{Total Cost} = (\text{False Positives} \times \text{Cost}_{\text{FP}}) + (\text{False Negatives} \times \text{Cost}_{\text{FN}})$$
  - *False Positive (FP)*: Predicting a candidate won't join when they actually will (leading to redundant backup sourcing costs).
  - *False Negative (FN)*: Predicting a candidate will join when they actually drop out (leading to vacant critical roles and lost project revenue).
- Automatically selects the threshold that yields the lowest economic impact.
