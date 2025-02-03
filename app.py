# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from config import CAR_DATA
from data_loader import load_dataset
from utils import get_nearest_age_subset, find_closest_cars
from guardrails import apply_guardrails
from predictors import predict_price_from_multiple_models

# Load dataset (will handle errors internally)
df = load_dataset()

from custom_transforms import log1p_transform

# # Inject the function into __main__ so that pickle can find it
# import __main__
# __main__.log1p_transform = log1p_transform



# -------------------------------
# Streamlit App Layout and Inputs
# -------------------------------
st.title("Used Car Price Prediction App with M-M-V Guardrails")
st.write("""
This application predicts the price of a used car based on various features.
**Guardrails** are applied per Make-Model-Variant (M-M-V) subset to clamp unrealistic values.
""")

# Sidebar Inputs
st.sidebar.header("Enter Car Details")
selected_make = st.sidebar.selectbox("Select Make", options=list(CAR_DATA.keys()))
selected_model = st.sidebar.selectbox("Select Model", options=list(CAR_DATA[selected_make].keys()))
selected_variant = st.sidebar.selectbox("Select Variant", options=CAR_DATA[selected_make][selected_model])
selected_city = st.sidebar.selectbox("City", [
    'Ahmedabad', 'Bangalore', 'Chennai', 'Gurgaon', 'Hyderabad', 'Kolkata',
    'Pune', 'Delhi', 'Panchkula', 'Ludhiana', 'Kharar', 'Coimbatore',
    'Noida', 'Ghaziabad', 'Lucknow', 'Mumbai', 'Thane', 'Mohali',
    'Kharagpur', 'Chandigarh', 'Ambala', 'Navi', 'Faridabad', 'Meerut',
    'Sangli', 'Surat', 'Mysore', 'Gulbarga', 'Ranga', 'Vadodara', 'Howrah'
])
selected_transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
selected_fuel_type = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
age = st.sidebar.number_input("Age (years)", min_value=0, max_value=50, value=5, step=1)
distance = st.sidebar.number_input("Odometer Reading (km)", min_value=0, max_value=500000, value=40000, step=1000)
range_percentage = st.sidebar.slider("Confidence Range (%)", 1, 20, 5)

# --------------
# Main Button: Prediction & Guardrails
# --------------
if st.button("Predict Price"):
    if not all([selected_make, selected_model, selected_variant, selected_city, selected_transmission, selected_fuel_type]):
        st.error("Please fill in all the car details.")
    else:
        # Get predictions from multiple models
        raw_predictions = predict_price_from_multiple_models(
            age=age,
            distance=distance,
            make=selected_make,
            car_model=selected_model,
            variant=selected_variant,
            city=selected_city,
            transmission=selected_transmission,
            fuel_type=selected_fuel_type
        )

        if not raw_predictions:
            st.error("An error occurred during prediction. Please check your inputs.")
        else:
            # Average the predictions
            avg_prediction = np.mean([p for p in raw_predictions.values() if p is not None])
            st.success(f"Average Predicted Price: ₹{round(avg_prediction)}")

            # Get M-M-V subset
            subset_mmv = get_nearest_age_subset(
                df=df,
                make=selected_make,
                model=selected_model,
                variant=selected_variant,
                age=age,
                min_samples=5,
                max_delta=5
            )

            if subset_mmv.empty:
                st.error("No data available for the selected Make-Model-Variant.")
            else:
                # Apply guardrails on the prediction
                guarded_price = apply_guardrails(
                    age=age,
                    distance=distance,
                    fuel_type=selected_fuel_type,
                    city=selected_city,
                    avg_prediction=avg_prediction,
                    df_subset=subset_mmv,
                    depreciation_rate=0.04,
                    min_floor=40000
                )

                if guarded_price is None:
                    st.error("Cannot predict a valid price under current regulations/constraints.")
                else:
                    st.success(f"Guarded Average Predicted Price: ₹{round(guarded_price)}")
                    lower_bound = guarded_price * (1 - range_percentage / 100)
                    upper_bound = guarded_price * (1 + range_percentage / 100)
                    st.write(f"Price Range (±{range_percentage}%): ₹{round(lower_bound)} - ₹{round(upper_bound)}")

                    similar_cars = find_closest_cars(selected_make, selected_model, selected_variant, age, distance, df)
                    if similar_cars.empty:
                        st.write("No similar cars found in the dataset for this M-M-V.")
                    else:
                        st.write("Closest Cars (based on Age & Odometer):")
                        st.dataframe(similar_cars)

                    # Optional Plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    mmv_subset = df[(df['Make'] == selected_make) & (df['Model'] == selected_model) & (df['Variant'] == selected_variant)]
                    ax.scatter(mmv_subset['Age'], mmv_subset['Price_numeric'], color='blue', alpha=0.5, label='Dataset Cars (Same M-M-V)')
                    ax.scatter(age, guarded_price, color='purple', s=200, marker='*', label='Guarded Average Prediction')
                    ax.fill_between([age - 1, age + 1], lower_bound, upper_bound, color='lightgreen', alpha=0.2, label=f"±{range_percentage}% Range")
                    ax.set_xlabel("Age (Years)")
                    ax.set_ylabel("Price (₹)")
                    ax.set_title(f"Age vs. Price for {selected_make} {selected_model} {selected_variant}")
                    ax.legend()
                    st.pyplot(fig)
