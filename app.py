
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow import keras

# -----------------------------------
# 1. Page configuration
# -----------------------------------
st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Bank Customer Churn Prediction")
st.write(
    "Enter the customer's details to predict whether "
    "they are likely to leave the bank."
)

# -----------------------------------
# 2. Load model and preprocessing files
# -----------------------------------
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(
        "model.h5",
        compile=False
    )

    with open("onehot_encoder_geo.pkl", "rb") as file:
        geo_encoder = pickle.load(file)

    with open("label_encoder_gender.pkl", "rb") as file:
        gender_encoder = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, geo_encoder, gender_encoder, scaler


try:
    model, geo_encoder, gender_encoder, scaler = load_artifacts()
except Exception as e:
    st.error(f"Could not load model or encoders: {e}")
    st.info(
        "Check that model.h5, onehot_encoder_geo.pkl, "
        "label_encoder_gender.pkl, and scaler.pkl exist "
        "in your project and use compatible library versions."
    )
    st.stop()

# -----------------------------------
# 3. Customer input form
# -----------------------------------
with st.form("customer_form"):

    st.subheader("Customer Information")

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=600
    )

    geography_options = list(geo_encoder.categories_[0])
    geography = st.selectbox(
        "Geography",
        geography_options
    )

    gender_options = list(gender_encoder.classes_)
    gender = st.selectbox(
        "Gender",
        gender_options
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40
    )

    tenure = st.number_input(
        "Tenure (years)",
        min_value=0,
        max_value=10,
        value=3
    )

    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=60000.0
    )

    num_products = st.selectbox(
        "Number of Products",
        [1, 2, 3, 4]
    )

    has_credit_card = st.selectbox(
        "Has Credit Card?",
        ["Yes", "No"]
    )

    is_active = st.selectbox(
        "Is Active Member?",
        ["Yes", "No"]
    )

    salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0
    )

    submitted = st.form_submit_button("Predict Churn")

# -----------------------------------
# 4. Process input and predict
# -----------------------------------
if submitted:
    try:
        # Encode gender using the fitted encoder
        gender_encoded = gender_encoder.transform([gender])[0]

        # Encode geography
        geo_result = geo_encoder.transform(
            pd.DataFrame({"Geography": [geography]})
        )

        # Handle sparse or dense encoder output
        if hasattr(geo_result, "toarray"):
            geo_result = geo_result.toarray()

        geo_columns = geo_encoder.get_feature_names_out(
            ["Geography"]
        )

        # Create customer features
        customer_data = pd.DataFrame([{
            "CreditScore": credit_score,
            "Gender": gender_encoded,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": 1 if has_credit_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active == "Yes" else 0,
            "EstimatedSalary": salary
        }])

        geography_data = pd.DataFrame(
            geo_result,
            columns=geo_columns
        )

        final_input = pd.concat(
            [customer_data, geography_data],
            axis=1
        )

        # Match the exact training feature order
        expected_columns = [
            "CreditScore",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "Geography_France",
            "Geography_Germany",
            "Geography_Spain"
        ]

        missing_columns = [
            col for col in expected_columns
            if col not in final_input.columns
        ]

        if missing_columns:
            st.error(
                f"Missing encoded features: {missing_columns}"
            )
            st.stop()

        final_input = final_input[expected_columns]

        # Scale features using the original fitted scaler
        scaled_input = scaler.transform(final_input)

        # Predict probability
        prediction = model.predict(
            scaled_input,
            verbose=0
        )

        probability = float(
            np.asarray(prediction).reshape(-1)[0]
        )

        st.subheader("Prediction Result")

        st.metric(
            "Customer Churn Probability",
            f"{probability:.2%}"
        )

        if probability >= 0.5:
            st.error(
                "⚠️ The model predicts that the customer "
                "is likely to leave the bank."
            )
        else:
            st.success(
                "✅ The model predicts that the customer "
                "is likely to stay with the bank."
            )

        st.caption(
            "This is a model prediction, not a guarantee "
            "of the customer's future behavior."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
