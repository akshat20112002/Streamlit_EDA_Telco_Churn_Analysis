# Telecom Customer Churn Analysis – End-to-End EDA with Streamlit

## Project Overview
This project presents an end-to-end exploratory data analysis (EDA) of a telecom customer churn dataset, implemented using **Streamlit** to deliver an interactive analytics dashboard. The objective is to analyze customer behavior, identify churn drivers, and generate actionable business insights to improve customer retention.

## Workflow Summary

### 1. Data Ingestion and Validation
- Dataset upload through the Streamlit interface  
- Schema inspection to verify feature structure  
- Missing value assessment and handling  
- Data type corrections:
  - `TotalCharges` converted from string to numeric
  - Invalid or non-parsable records removed  
- Customer identifier columns dropped to avoid analytical noise

### 2. Feature Segmentation
- Separation of features into:
  - **Categorical variables**
  - **Numerical variables**
- Enables appropriate statistical and visualization techniques for each feature type

### 3. Exploratory Data Analysis

#### Univariate Analysis
- Distribution analysis of individual features  
- Frequency plots for categorical variables  
- Histograms and summary statistics for numerical variables  

#### Bivariate Analysis
- Relationship analysis between features and customer churn  
- Visualization techniques:
  - Bar charts for categorical features vs. churn
  - Box plots for numerical features vs. churn  

#### Correlation Analysis
- Correlation matrix computed for numerical variables  
- Identification of linear relationships and potential multicollinearity  

### 4. Feature Engineering
Business-driven derived features created to enhance analytical depth:
- **Tenure Groups** (short-term, mid-term, long-term customers)
- **Long-Term Contract Indicator**
- **Family Association** (partner and dependents combined)
- **Multi-Product Usage** (count of subscribed services)
- **Electronic Payment Indicator**

### 5. Statistical Validation
To ensure analytical rigor, statistical hypothesis testing is applied:
- **T-tests** for numerical features to compare churned vs. retained customers
- **Chi-square tests** for categorical features to assess association with churn  
- Significance testing used to validate observed churn patterns

### 6. Segmentation and Insights
- Customer segmentation performed to identify high-risk churn groups  
- Key behavioral and contractual patterns highlighted  

### 7. Business Recommendations
- Actionable insights derived from analysis
- Recommendations focused on:
  - Retention strategies for high-risk segments
  - Contract optimization
  - Targeted engagement based on customer behavior

## Tools and Technologies
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- SciPy, Statsmodels  
- Streamlit  

## Outcome
The project delivers an interactive Streamlit dashboard that combines statistical validation, business-oriented feature engineering, and visual analytics to support data-driven decision-making for telecom customer retention.

Project Link: https://telco-customer-analyzer.streamlit.app/
