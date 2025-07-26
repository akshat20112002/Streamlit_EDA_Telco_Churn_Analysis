import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Telco Customer Churn EDA", layout="wide")

st.title("Telco Customer Churn - Comprehensive EDA")

# File uploader to load data
uploaded_file = st.file_uploader("Upload your Telco Customer Churn CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Data loaded successfully!")

        # 2. Basic Data Overview
        st.header("1. Basic Data Overview")
        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.dataframe(df.head(3))
        st.write("Data Types:")
        st.write(df.dtypes)
        st.write("Null values per column:")
        st.write(df.isnull().sum())

        # 3. Data Cleaning
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(subset=['TotalCharges'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        if 'customerID' in df.columns:
            df.drop('customerID', axis=1, inplace=True)

        # 4. Setup for EDA
        cat_cols = [col for col in df.select_dtypes('object').columns if col != 'Churn']
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # 5. General Statistics
        st.header("2. General Statistics")
        churn_dist = df['Churn'].value_counts(normalize=True)
        st.subheader("Churn Distribution")
        st.bar_chart(churn_dist)
        st.write("Numeric Summary:")
        st.write(df[num_cols].describe())

        # 6. Univariate Analysis
        st.header("3. Univariate Analysis")
        st.subheader("Categorical Features")
        for col in ['Churn'] + cat_cols:
            fig, ax = plt.subplots(figsize=(6,2))
            sns.countplot(x=col, data=df, palette='pastel', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        st.subheader("Numerical Features")
        fig, ax = plt.subplots(figsize=(12,4))
        df[num_cols].hist(bins=20, ax=ax)
        st.pyplot(fig)

        # 7. Bivariate Analysis
        st.header("4. Bivariate Analysis")
        st.subheader("Categorical vs. Churn")
        for col in cat_cols:
            ctab = pd.crosstab(df[col], df['Churn'], normalize='index')
            st.bar_chart(ctab)

        st.subheader("Numeric vs. Churn")
        for col in num_cols:
            fig, ax = plt.subplots(figsize=(6,2))
            sns.boxplot(x='Churn', y=col, data=df, palette='Set3', ax=ax)
            st.pyplot(fig)

        # 8. Correlation Analysis
        st.header("5. Correlation Analysis")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap='Blues', ax=ax)
        st.pyplot(fig)

        # 9. Feature Engineering and Derived Features
        st.header("6. Feature Engineering and Derived Features")
        df['TenureGroup'] = pd.cut(df['tenure'], bins=[0,12,36,72], labels=['New','Mature','Loyal'], right=True)
        df['MultiProducts'] = ((df['PhoneService']=='Yes') & (df['InternetService']!='No')).astype(int)
        df['HasFamily'] = ((df['Partner']=='Yes') | (df['Dependents']=='Yes')).astype(int)
        df['LongTerm'] = (df['Contract'] == 'Two year').astype(int)
        df['ElectronicPay'] = df['PaymentMethod'].str.lower().str.contains('electronic').astype(int)

        for col in ['TenureGroup']:
            fig, ax = plt.subplots(figsize=(6,2))
            sns.countplot(x=col, data=df, hue='Churn', palette='muted', ax=ax)
            st.pyplot(fig)
        for col in ['MultiProducts', 'HasFamily', 'LongTerm', 'ElectronicPay']:
            fig, ax = plt.subplots(figsize=(6,2))
            sns.barplot(x=col, y=df['Churn'].map({'No':0, 'Yes':1}), data=df, ci=None, palette='cool', ax=ax)
            st.pyplot(fig)

        # 10. Pairwise Plots (optional)
        st.header("7. Pairwise Plots (Numeric Features)")
        st.info("Tip: For large datasets, this may take a while.")
        if st.button("Show Pairplot"):
            import warnings
            warnings.filterwarnings("ignore")
            sns.pairplot(df, vars=['tenure','MonthlyCharges','TotalCharges'], hue='Churn', plot_kws={"alpha":0.5}, diag_kind='kde')
            st.pyplot(plt.gcf())

        # 11. Statistical Tests
        st.header("8. Statistical Tests")
        from scipy.stats import ttest_ind, chi2_contingency

        st.subheader("T-Test Results (Numeric Features)")
        ttest_results = []
        for col in num_cols:
            try:
                group1 = df[df['Churn']=='Yes'][col]
                group2 = df[df['Churn']=='No'][col]
                stat, p = ttest_ind(group1, group2)
                ttest_results.append({'Feature': col, 't-stat': stat, 'p-value': p})
            except Exception as e:
                ttest_results.append({'Feature': col, 't-stat': 'ERROR', 'p-value': str(e)})
        st.dataframe(pd.DataFrame(ttest_results))

        st.subheader("Chi2 Test Results (Categorical Features)")
        chi2_results = []
        for col in cat_cols:
            try:
                table = pd.crosstab(df[col], df['Churn'])
                stat, p, dof, expected = chi2_contingency(table)
                chi2_results.append({'Feature': col, 'Chi2': stat, 'p-value': p})
            except Exception as e:
                chi2_results.append({'Feature': col, 'Chi2': 'ERROR', 'p-value': str(e)})
        st.dataframe(pd.DataFrame(chi2_results))

        # 12. Segmentation Plots
        st.header("9. Segmentation Plots")
        fig, axes = plt.subplots(1,2, figsize=(14,4))
        sns.barplot(x='TenureGroup', y=df['Churn'].map({'No':0, 'Yes':1}),
                    hue='Contract', data=df, ax=axes[0], palette='Set2')
        axes[0].set_title('Churn Rate by Tenure Group & Contract')
        sns.barplot(x='InternetService', y=df['Churn'].map({'No':0, 'Yes':1}),
                    data=df, ax=axes[1], palette='Set2')
        axes[1].set_title('Churn Rate by Internet Service')
        st.pyplot(fig)

        # 13. Insights & Recommendations
        st.header("10. Insights & Business Recommendations")
        st.markdown("""
        - **Month-to-month contracts** show highest churn rates.
        - **High monthly charges** and **low tenure** strongly correlate with churn.
        - Customers **without family ties** (Partner/Dependents) are at greater risk.
        - **Electronic payment** users may be less engaged, increasing churn risk.
        - **Focus retention efforts** on new, month-to-month, and high-charge segments.
        """)
    except Exception as e:
        st.error(f"Error loading or processing file: {e}")
else:
    st.warning("Please upload your data file to proceed.")
