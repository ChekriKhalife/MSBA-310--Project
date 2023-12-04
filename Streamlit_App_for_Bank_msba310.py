import streamlit as st
import numpy as np
import pandas as pd

# Set page layout
st.set_page_config(page_title="Logistic Regression Prediction", layout="wide")
# Custom CSS
st.markdown("""
<style>
.sidebar .sidebar-content {
    background-color: #f0f2f6;
}
.stButton>button {
    width: 100%;
    border-radius: 5px;
    background-color: #0e1117;
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("Predictive Analysis using Logistic Regression")
st.markdown("""
    This application uses a logistic regression model to predict the outcome based on the provided inputs.
""")

# Logistic Regression Coefficients

coef = {
    "intercept": -2.62182835,
    "job_blue_collar": -0.28880525,
    "job_entrepreneur": -0.34575561,
    "job_housemaid": -0.44709644,
    "job_management": -0.14657952,
    "job_retired": 0.28961015,
    "job_self_employed": -0.20036027,
    "job_services": -0.20294890,
    "job_student": 0.41141637,
    "job_technician": -0.14363973,
    "job_unemployed": -0.13365994,
    "job_unknown": -0.18481084,
    "marital_married": -0.23894135,
    "marital_single": 0.01204454,
    "education_secondary": 0.20780991,
    "education_tertiary": 0.41078565,
    "education_unknown": 0.21265528,
    "housing_yes": -0.62755908,
    "loan_yes": -0.41950486,
    "contact_telephone": -0.12645797,
    "contact_unknown": -1.56251050,
    "day": 0.00993415,
    "month_aug": -0.62001697,
    "month_dec": 0.75773672,
    "month_feb": -0.12741633,
    "month_jan": -1.25830236,
    "month_jul": -0.81197776,
    "month_jun": 0.47683364,
    "month_mar": 1.60070269,
    "month_may": -0.45444435,
    "month_nov": -0.89490039,
    "month_oct": 0.86920519,
    "month_sep": 0.85300411,
    "duration": 0.00420852,  
    "campaign": -0.08815015,  
    "previous": 0.03258026,  
    "poutcome_other": 0.26508658,
    "poutcome_success": 2.28493062,
    "poutcome_unknown": -0.01619976
}

# Function for individual predictions
def individual_prediction():
    col1, col2, col3 = st.columns(3)

    with col1:
        job = st.selectbox('Job', ['admin', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
        marital = st.selectbox('Marital Status', ['married', 'single', 'divorced'])
        education = st.selectbox('Education Level', ['primary', 'secondary', 'tertiary', 'unknown'])

    with col2:
        housing = st.selectbox('Housing Loan', ['yes', 'no'])
        loan = st.selectbox('Personal Loan', ['yes', 'no'])
        contact = st.selectbox('Contact Communication Type', ['telephone', 'unknown', 'cellular'])
        month = st.selectbox('Month of last contact', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])

    with col3:
        poutcome = st.selectbox('Previous Campaign Outcome', ['other', 'success', 'unknown', 'failure'])
        day = st.number_input('Day of the month of the last contact', min_value=1, max_value=31)
        duration = st.number_input('Duration of last contact (s)', min_value=0)
        campaign = st.number_input('Campaign Contacts Count', min_value=0)
        previous = st.number_input('Previous: Number of days passed since previously contacted', min_value=0)

    if st.button('Predict'):
    # Mapping user inputs to coefficients
        input_values = {
        "job_entrepreneur": 1 if job == "entrepreneur" else 0,
        "job_housemaid": 1 if job == "housemaid" else 0,
        "job_management": 1 if job == "management" else 0,
        "job_retired": 1 if job == "retired" else 0,
        "job_self_employed": 1 if job == "self-employed" else 0,
        "job_services": 1 if job == "services" else 0,
        "job_student": 1 if job == "student" else 0,
        "job_technician": 1 if job == "technician" else 0,
        "job_unemployed": 1 if job == "unemployed" else 0,
        "job_unknown": 1 if job == "unknown" else 0,
        "marital_divorced": 1 if marital == "divorced" else 0,
        "marital_single": 1 if marital == "single" else 0,
        "education_primary": 1 if education == "primary" else 0,
        "education_tertiary": 1 if education == "tertiary" else 0,
        "education_unknown": 1 if education == "unknown" else 0,
        "housing_yes": 1 if housing == "yes" else 0,
        "loan_yes": 1 if loan == "yes" else 0,
        "contact_telephone": 1 if contact == "telephone" else 0,
        "contact_unknown": 1 if contact == "unknown" else 0,
        "month_aug": 1 if month == "aug" else 0,
        "month_dec": 1 if month == "dec" else 0,
        "month_feb": 1 if month == "feb" else 0,
        "month_jan": 1 if month == "jan" else 0,
        "month_jul": 1 if month == "jul" else 0,
        "month_jun": 1 if month == "jun" else 0,
        "month_mar": 1 if month == "mar" else 0,
        "month_apr": 1 if month == "may" else 0,
        "month_nov": 1 if month == "nov" else 0,
        "month_oct": 1 if month == "oct" else 0,
        "month_sep": 1 if month == "sep" else 0,
        "day": int(day),  # Numerical value of the day
        "duration": duration,
        "campaign": campaign,
        "previous": previous,
        "poutcome_other": 1 if poutcome == "other" else 0,
        "poutcome_success": 1 if poutcome == "success" else 0,
        "poutcome_failure": 1 if poutcome == "failure" else 0,
    }

        # Compute logistic regression score
        score = coef["intercept"]
        for key, value in coef.items():
            if key != "intercept":
                score += value * input_values.get(key, 0)

        # Apply logistic function
        probability = 1 / (1 + np.exp(-score))

        # Apply threshold
        prediction = 'Yes' if probability > 0.15 else 'No'

        # Display the result in a visually distinct section
        st.markdown('---')
        st.markdown('## 📊 Prediction Result')
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="Prediction", value=prediction)
      
# Function to predict using logistic regression
def logistic_regression_prediction(row, coefficients):
    input_values = {
        "job_" + row['Job']: 1,
        "marital_" + row['Marital Status']: 1,
        "education_" + row['Education Level']: 1,
        "housing_yes": 1 if row['Housing Loan'] == 'yes' else 0,
        "loan_yes": 1 if row['Personal Loan'] == 'yes' else 0,
        "contact_" + row['Contact Communication Type']: 1,
        "month_" + row['Month of last contact']: 1,
        "day": row['Day of the month of the last contact'],
        "duration": row['Duration of last contact (s)'],
        "campaign": row['Campaign Contacts Count'],
        "previous": row['Previous: Number of days passed since previously contacted'],
        "poutcome_" + row['Previous Campaign Outcome']: 1
    }

    # Compute logistic regression score
    score = coefficients["intercept"]
    for key, value in coefficients.items():
        if key != "intercept":
            score += value * input_values.get(key, 0)

    # Apply logistic function
    probability = 1 / (1 + np.exp(-score))

    # Apply threshold
    prediction = 'Yes' if probability > 0.15 else 'No'

    return probability, prediction

# Function for bulk predictions
def bulk_prediction():
    st.subheader("Bulk Upload for Predictions")
    uploaded_file = st.file_uploader("Upload your input Excel file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read the Excel file into a pandas DataFrame
            df = pd.read_csv(uploaded_file)

            # Predict for each row in the DataFrame
            predictions = df.apply(lambda x: logistic_regression_prediction(x, coef), axis=1)
            
            # Add predictions to DataFrame
            df['Predicted Probability'], df['Prediction'] = zip(*predictions)

            # Display results
            st.subheader("Predictions")
            st.write(df)

            # Filter and display target individuals
            target_df = df[df['Prediction'] == 'Yes'][['ID', 'Name', 'Phone Number']]
            if not target_df.empty:
                st.markdown('---')
                st.markdown('## 🎯 People to Target')
                st.write(target_df.to_html(index=False), unsafe_allow_html=True)  # Display DataFrame without index
                
                # Download link for filtered DataFrame
                csv = target_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='target_individuals.csv',
                    mime='text/csv',
                )
            else:
                st.markdown('---')
                st.markdown('## 🎯 No Target Recommendations')
                st.markdown('Based on the predictions, there are no individuals to target at this time.')
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Sidebar for prediction type selection
prediction_type = st.sidebar.radio(
    "Choose Prediction Type",
    ('Individual Prediction', 'Bulk Prediction', 'About the Model')
)
# Main layout based on prediction type
if prediction_type == 'Individual Prediction':
    individual_prediction()
elif prediction_type == 'Bulk Prediction':
    bulk_prediction()
elif prediction_type == 'About the Model':
    st.markdown("""
    ## Model Information
    - This model uses logistic regression to predict outcomes based on several factors.
    - It calculates the probability and classifies the output as Yes or No based on a threshold value.
    """)

