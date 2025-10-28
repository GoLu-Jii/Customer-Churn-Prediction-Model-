# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib

# --- Redefine feature engineering function ---
def feature_engineering_fn(df):
    df = df.copy()
    df['is_auto_payment'] = df['PaymentMethod'].apply(
        lambda x: 1 if 'automatic' in x.lower() else 0
    )
    df['has_streaming'] = df[['StreamingTV', 'StreamingMovies']].apply(
        lambda x: 1 if 'Yes' in x.values else 0, axis=1
    )
    df['has_support_services'] = df[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']].apply(
        lambda x: 1 if 'Yes' in x.values else 0, axis=1
    )
    df['avg_monthly_from_total'] = pd.to_numeric(df['TotalCharges'], errors='coerce') / (df['tenure'] + 1)
    df['tenure_per_monthly'] = df['tenure'] / (df['MonthlyCharges'] + 1)
    return df


# --- Load saved pipeline ---
@st.cache_resource
def load_model():
    return joblib.load("final_churn_xgb_pipeline.pkl")

model_pipeline = load_model()


# --- App Header ---
st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict whether they are likely to churn.")

# --- User Input ---
with st.form("input_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.text_input("Total Charges", "2000")

    submit = st.form_submit_button("Predict")

# --- Prediction ---
if submit:
    new_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])

    # Apply feature engineering
    new_data = feature_engineering_fn(new_data)

    # Predict
    pred = model_pipeline.predict(new_data)[0]
    prob = model_pipeline.predict_proba(new_data)[0, 1]

    st.subheader("🔍 Prediction Result")
    if pred == 1:
        st.error(f"⚠️ This customer is **likely to churn** (Probability: {prob:.2f})")
    else:
        st.success(f"✅ This customer is **not likely to churn** (Probability: {prob:.2f})")
