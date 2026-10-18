
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# -----------------------------------
# 1. Load trained model and encoders
# -----------------------------------

@st.cache_resource
def load_artifacts():
    model = load_model("model.h5")

    with open("onehot_encoder_geo.pkl", "rb") as file:
        onehot_encoder_geo = pickle.load(file)

    with open("label_encoder_gender.pkl", "rb") as file:
        label_encoder_gender = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, onehot_encoder_geo, label_encoder_gender, scaler


try:
    model, onehot_encoder_geo, label_encoder_gender, scaler = load_artifacts()
except Exception as e:
    st.error(f"Could not load model or pickle files: {e}")
    st.stop()


# -----------------------------------
# 2. Streamlit interface
# -----------------------------------

st.title("Customer Churn Prediction")
st.write(
    "Enter customer details to predict whether the customer may leave the bank."
)

# Use the categories stored in the trained Geography encoder
geography_options = list(onehot_encoder_geo.categories_[0])

# Use a standard dropdown if the saved gender encoder is fitted.
# Otherwise, use the common training categories.
if hasattr(label_encoder_gender, "classes_"):
    gender_options = list(label_encoder_gender.classes_)
else:
    gender_options = ["Female", "Male"]

credit_score = st.number_input(
    "Credit Score", min_value=300, max_value=900, value=600
)

geography = st.selectbox("Geography", geography_options)

gender = st.selectbox("Gender", gender_options)

age = st.number_input(
    "Age", min_value=18, max_value=100, value=40
)

tenure = st.number_input(
    "Tenure (years)", min_value=0, max_value=10, value=3
)

balance = st.number_input(
    "Account Balance", min_value=0.0, value=60000.0
)

num_of_products = st.selectbox(
    "Number of Products", [1, 2, 3, 4]
)

has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])

is_active_member = st.selectbox(
    "Is Active Member?", ["Yes", "No"]
)

estimated_salary = st.number_input(
    "Estimated Salary", min_value=0.0, value=50000.0
)


# -----------------------------------
# 3. Predict churn
# -----------------------------------

if st.button("Predict Churn"):

    try:
        # Encode Gender
        if hasattr(label_encoder_gender, "classes_"):
            gender_encoded = label_encoder_gender.transform([gender])[0]
        else:
            st.error(
                "Your gender encoder pickle is not fitted. "
                "Fix label_encoder_gender.pkl using your original "
                "training notebook before predicting."
            )
            st.stop()

        # Encode Geography using the fitted OneHotEncoder
        geo_encoded = onehot_encoder_geo.transform(
            pd.DataFrame({"Geography": [geography]})
        ).toarray()

        geo_columns = onehot_encoder_geo.get_feature_names_out(
            ["Geography"]
        )

        # Prepare customer features
        customer_data = pd.DataFrame([{
            "CreditScore": credit_score,
            "Gender": gender_encoded,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_of_products,
            "HasCrCard": 1 if has_cr_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active_member == "Yes" else 0,
            "EstimatedSalary": estimated_salary
        }])

        # Add Geography encoding
        geography_data = pd.DataFrame(
            geo_encoded, columns=geo_columns
        )

        final_input = pd.concat(
            [customer_data, geography_data], axis=1
        )

        # Match the exact feature order used during model training.
        # This list assumes the common 12-feature churn model.
        expected_columns = [
            "CreditScore", "Gender", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember",
            "EstimatedSalary", "Geography_France",
            "Geography_Germany", "Geography_Spain"
        ]

        if set(final_input.columns) != set(expected_columns):
            st.error(
                "Input columns do not match the expected model features. "
                "Check your original training notebook and encoder categories."
            )
            st.stop()

        final_input = final_input[expected_columns]

        # Scale features
        scaled_input = scaler.transform(final_input)

        # Predict
        prediction = model.predict(scaled_input, verbose=0)
        probability = float(np.asarray(prediction).reshape(-1)[0])

        st.subheader("Prediction Result")
        st.metric("Customer Churn Probability", f"{probability:.2%}")

        if probability >= 0.5:
            st.error("The customer is likely to leave the bank.")
        else:
            st.success("The customer is likely to stay with the bank.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
