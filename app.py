import streamlit as st
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pickle

class CreditWiseLoanSystem:
    def __init__(self):
        st.set_page_config(page_title="Credit Wise Loan System", layout="wide")
        st.title("Credit Wise Loan System")

        self.collect_inputs()

        if st.button("Submit Application"):
            self.predict_loan_approval()

    def collect_inputs(self):
        st.subheader("Applicant Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            self.applicant_income = st.number_input(
                "Applicant Income", min_value=0, step=1000
            )
            self.coapplicant_income = st.number_input(
                "Coapplicant Income", min_value=0, step=1000
            )
            self.age = st.number_input(
                "Age", min_value=18, max_value=75
            )
            self.gender = st.selectbox(
                "Gender", ["Male", "Female"]
            )

        with col2:
            self.marital_status = st.selectbox(
                "Marital Status", ["Single", "Married"]
            )
            self.dependents = st.number_input(
                "Number of Dependents", min_value=0, max_value=10
            )
            self.education_level = st.selectbox(
                "Education Level", ["Graduate", "Not Graduate"]
            )
            self.employment_status = st.selectbox(
                "Employment Status",
                ["Salaried", "Self-employed", "Contract", "Unemployed"]
            )

        with col3:
            self.employer_category = st.selectbox(
                "Employer Category", ["Private", "Government", "MNC", "Unemployed"]
            )
            self.credit_score = st.number_input(
                "Credit Score", min_value=300, max_value=900
            )
            self.existing_loans = st.number_input(
                "Existing Loans", min_value=0
            )

        st.subheader("Financial & Loan Details")

        col4, col5, col6 = st.columns(3)

        with col4:
            self.dti_ratio = st.number_input(
                "DTI Ratio (%)", min_value=0.0, max_value=100.0
            )
            self.savings = st.number_input(
                "Savings Amount", min_value=0
            )

        with col5:
            self.collateral_value = st.number_input(
                "Collateral Value", min_value=0
            )
            self.loan_amount = st.number_input(
                "Loan Amount", min_value=0
            )

        with col6:
            self.loan_term = st.number_input(
                "Loan Term (Months)", min_value=6
            )
            self.loan_purpose = st.selectbox(
                "Loan Purpose",
                ["Personal", "Car", "Business", "Home", "Education"]
            )
            self.property_area = st.selectbox(
                "Property Area",
                ["Urban", "Semiurban", "Rural"]
            )

    def predict_loan_approval(self):

        input_df = pd.DataFrame([{
            "Applicant_Income": self.applicant_income,
            "Coapplicant_Income": self.coapplicant_income,
            "Age": self.age,
            "Gender": self.gender,
            "Marital_Status": self.marital_status,
            "Dependents": self.dependents,
            "Education_Level": self.education_level,
            "Employment_Status": self.employment_status,
            "Employer_Category": self.employer_category,
            "Credit_Score": self.credit_score,
            "Existing_Loans": self.existing_loans,
            "DTI_Ratio": self.dti_ratio,
            "Savings": self.savings,
            "Collateral_Value": self.collateral_value,
            "Loan_Amount": self.loan_amount,
            "Loan_Term": self.loan_term,
            "Loan_Purpose": self.loan_purpose,
            "Property_Area": self.property_area
        }])

        with open("model/loan_pipeline.pkl", "rb") as file:
            pipeline = pickle.load(file)

        prediction = pipeline.predict(input_df)[0]

        st.subheader("Loan Approval Result")

        if prediction == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Not Approved")

if __name__ == "__main__":
    CreditWiseLoanSystem()
