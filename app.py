import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1) Define Custom Transform Function
# -----------------------------------------------------------------------------
def log1p_transform(x):
    """
    Custom log1p transform function used in the model's preprocessing pipeline.
    """
    return np.log1p(x)

# -----------------------------------------------------------------------------
# 2) Load the Model and Dataset
# -----------------------------------------------------------------------------
MODEL_PATH = 'hybrid_model_for_hyundai_and_suzuki.joblib'
DATASET_PATH = 'refrence_data.csv'

# Load the model with proper error handling
try:
    model = joblib.load(MODEL_PATH)
    if isinstance(model, str):
        raise ValueError("Loaded model is a string. Please check the model file.")
except AttributeError as e:
    st.error(f"Error loading the model: {e}")
    st.stop()
except FileNotFoundError:
    st.error(f"Model file not found at path: {MODEL_PATH}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the model: {e}")
    st.stop()

# Load the dataset with proper error handling
try:
    df = pd.read_csv(DATASET_PATH)
    required_columns = {'Make', 'Model', 'Variant', 'Age', 'Distance_numeric', 'Price_numeric'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        st.error(f"The dataset is missing required columns: {missing}")
        st.stop()
except FileNotFoundError:
    st.error(f"Dataset file not found at path: {DATASET_PATH}")
    st.stop()
except pd.errors.EmptyDataError:
    st.error("The dataset file is empty.")
    st.stop()
except pd.errors.ParserError:
    st.error("Error parsing the dataset file. Please check its format.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the dataset: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3) Helper Functions
# -----------------------------------------------------------------------------
def get_age_bracket(age):
    """
    Returns a label based on the age bucket.
    """
    if age < 3:
        return '0-3'
    elif age < 5:
        return '3-5'
    elif age < 10:
        return '5-10'
    elif age < 15:
        return '10-15'
    else:
        return '15+'

def get_mileage_bracket(mileage):
    """
    Returns a label based on the odometer bucket.
    """
    if mileage < 30000:
        return '0-30k'
    elif mileage < 60000:
        return '30k-60k'
    elif mileage < 100000:
        return '60k-100k'
    elif mileage < 150000:
        return '100k-150k'
    else:
        return '150k+'

def get_nearest_age_subset(df, make, model, variant, age, min_samples=5, max_delta=5):
    """
    Selects a subset of the dataframe based on the nearest age group.
    Starts with the same age, then expands the range by ±1, ±2, etc., until
    at least min_samples are found or max_delta is reached.
    
    Parameters:
        df (pd.DataFrame): The entire dataset.
        make (str): Car make.
        model (str): Car model.
        variant (str): Car variant.
        age (int): Age of the car.
        min_samples (int): Minimum number of samples required.
        max_delta (int): Maximum range to expand the age difference.
    
    Returns:
        pd.DataFrame: Subset of the dataframe based on the nearest age group.
    """
    subset = df[
        (df['Make'] == make) &
        (df['Model'] == model) &
        (df['Variant'] == variant)
    ]

    if subset.empty:
        return subset  # Empty subset

    max_age = subset['Age'].max()

    # If input age exceeds max_age, return the entire subset to handle depreciation
    if age > max_age:
        return subset

    # Initialize delta
    delta = 0
    while delta <= max_delta:
        age_min = age - delta
        age_max = age + delta
        # Ensure age_min is not negative
        age_min = max(age_min, 0)
        subset_age = subset[
            (subset['Age'] >= age_min) &
            (subset['Age'] <= age_max)
        ]
        if len(subset_age) >= min_samples:
            return subset_age
        delta += 1

    # If not enough samples even after max_delta, return the closest possible subset
    return subset[
        (subset['Age'] >= max(age - max_delta, 0)) &
        (subset['Age'] <= min(age + max_delta, max_age))
    ]

def apply_guardrails(age, distance, fuel_type, city, raw_prediction, df_subset, depreciation_rate=0.02, min_floor=50000):
    """
    Applies post-prediction guardrails based on:
    1) Extreme regulatory constraints (age + fuel + city).
    2) Subset-based percentile clamping (from the M-M-V subset).
    3) Depreciation logic for ages beyond the dataset.
    
    Returns:
       - final_price (float) if valid
       - None if we disclaim "Cannot Predict"
    """
    # -------------------------------
    # 1) Check Regulatory Constraints
    # -------------------------------
    if (fuel_type.lower() == 'diesel') and (age > 10) and (city.lower() == 'delhi'):
        return None  # Disclaim

    if (fuel_type.lower() == 'petrol') and (age > 15) and (city.lower() == 'delhi'):
        return None  # Disclaim

    # -------------------------------
    # 2) Determine Maximum Age in Subset
    # -------------------------------
    max_age_in_subset = df_subset['Age'].max()
    price_at_max_age = df_subset[df_subset['Age'] == max_age_in_subset]['Price_numeric'].mean()

    # -------------------------------
    # 3) Apply Clamping or Depreciation
    # -------------------------------
    if age <= max_age_in_subset:
        # Age within the dataset range: Apply percentile clamping
        if len(df_subset) >= 5:  # Require at least 5 samples for reliable percentiles
            p5, p95 = np.percentile(df_subset['Price_numeric'], [5, 95])
            lower_bound = p5 * 0.95
            upper_bound = p95 * 1.05
            if lower_bound <= raw_prediction <= upper_bound:
                clamped_price = raw_prediction
            else:
                clamped_price = (p5 + p95) / 2
        else:
            # Fallback if the subset is too small
            clamped_price = max(raw_prediction, min_floor)
    else:
        # Age exceeds the dataset range: Apply depreciation
        years_beyond = age - max_age_in_subset
        depreciated_price = price_at_max_age * ((1 - depreciation_rate) ** years_beyond)
        clamped_price = depreciated_price

    # -------------------------------
    # 4) Enforce Minimum Floor
    # -------------------------------
    final_price = max(clamped_price, min_floor)
    return final_price

def find_closest_cars(make, model, variant, age, distance, df):
    """
    Returns up to 5 closest cars in the dataset based on the same M-M-V.
    """
    filtered = df[
        (df['Make'] == make) &
        (df['Model'] == model) &
        (df['Variant'] == variant)
    ]

    if filtered.empty:
        return pd.DataFrame()

    filtered = filtered.copy()
    filtered['Age_Diff'] = abs(filtered['Age'] - age)
    filtered['Odom_Diff'] = abs(filtered['Distance_numeric'] - distance)
    closest_cars = filtered.sort_values(by=['Age_Diff', 'Odom_Diff']).head(5)
    return closest_cars[['Make', 'Model', 'Variant', 'Age', 'Distance_numeric', 'Price_numeric']]

# -----------------------------------------------------------------------------
# 4) Streamlit Application Layout and Inputs
# -----------------------------------------------------------------------------
st.title("Used Car Price Prediction App (M-M-V Guardrails)")
st.write("""
This application predicts the price of a used car based on various features.
**Guardrails** are derived **per Make-Model-Variant (M-M-V)** subset to clamp unrealistic values.
For cars older than the dataset's maximum age for their M-M-V, a constant depreciation rate is applied.
""")

# Sidebar Inputs
st.sidebar.header("Enter the Car Features")


d = {'Hyundai': {'i20 Active': ['1.2 S'],
  'Creta': ['1.4 S[2015-2017]',
   '1.6 S Petrol[2015-2017]',
   '1.6 SX[2015-2017]',
   'E Plus 1.4 CRDI[2017-2018]',
   'E Plus 1.6 Petrol[2017-2018]',
   'SX (O) 1.4 Turbo 7 DCT[2020-2023]',
   'SX 1.5 Diesel[2020-2023]',
   'SX 1.6 AT CRDi[2018-2019]',
   'SX 1.6 AT Petrol[2018-2019]',
   'SX 1.6 CRDi[2018-2019]',
   'SX 1.6 Petrol[2018-2019]',
   'SX Plus 1.6  Petrol[2017-2018]',
   'SX Plus 1.6 AT CRDI[2017-2018]',
   '1.6 SX Plus AT Petrol[2015-2017]'],
  'Elite i20': ['Asta 1.2[2017-2018]',
   'Asta 1.2[2014-2015]',
   'Magna 1.2[2014-2015]',
   'Sportz 1.2[2017-2018]',
   'Sportz 1.2[2019-2020]',
   'Sportz 1.2[2014-2015]'],
  'Grand i10': ['Asta 1.2 Kappa VTVT',
   'Asta AT 1.2 Kappa VTVT (O) [2016-2017][2013-2017]',
   'Asta AT 1.2 Kappa VTVT [2013-2016][2013-2017]',
   'Magna 1.2 Kappa VTVT',
   'Sports Edition 1.2L Kappa VTVT[2013-2017]',
   'Sportz (O) 1.2 Kappa VTVT [2017-2018]',
   'Sportz (O) AT 1.2 Kappa VTVT [2017-2018]',
   'Sportz 1.2 Kappa VTVT'],
  'Xcent': ['E'],
  'Eon': ['Era +', 'Magna +'],
  'Santro': ['Magna', 'Sportz'],
  'i20': ['Magna 1.2[2012-2014]', 'Asta 1.2 MT', 'Sportz 1.2 MT'],
  'Venue': ['S 1.2 Petrol',
   'SX 1.0 Turbo',
   'SX 1.5 CRDi',
   'SX Plus 1.0 Turbo DCT'],
  'Aura': ['S 1.2 CNG'],
  'i10': ['Sportz 1.1 iRDE2 [2010--2017][2010-2017]', 'Sportz 1.2[2007-2010]'],
  'Verna': ['i[2006-2010]']},
 'Maruti Suzuki': {'Eeco': ['5 STR[2010-2022]', '5 STR AC'],
  'Baleno': ['Alpha[2019-2022]',
   'Alpha 1.2[2015-2019]',
   'Delta 1.2[2015-2019]',
   'Delta 1.3[2015-2019]',
   'Delta MT',
   'Zeta[2019-2022]',
   'Zeta 1.2[2015-2019]'],
  'Ciaz': ['Alpha 1.4 AT[2017-2018]',
   'Alpha 1.4 MT[2017-2018]',
   'Alpha Hybrid 1.5 AT [2018-2020]',
   'Alpha Hybrid 1.5 [2018-2020]',
   'VDi+ SHVS[2014-2017]',
   'VXi[2014-2017]',
   'ZDi+ SHVS[2014-2017]',
   'ZXI+[2014-2017]',
   'ZXi[2014-2017]',
   'Zeta 1.4 AT[2017-2018]'],
  'XL6': ['Alpha AT Petrol', 'Alpha MT Petrol', 'Zeta MT Petrol'],
  'Ignis': ['Delta 1.2 MT', 'Zeta 1.2 AMT'],
  'Swift DZire': ['LDI[2011-2015]', 'VDI[2011-2015]', 'VXI[2011-2015]'],
  'Vitara Brezza': ['VDi[2016-2020]', 'ZDi[2016-2020]'],
  'Swift': ['LXi',
   'VDi[2014-2018]',
   'VXi',
   'ZDi[2014-2018]',
   'ZXi',
   'VXi AMT',
   'VDi[2011-2014]'],
  'Alto 800': ['Lx[2012-2016]', 'VXi', 'LXi'],
  'Ertiga': ['VDI SHVS[2015-2018]',
   'VDi[2012-2015]',
   'VXi[2018-2022]',
   'VXI[2015-2018]',
   'ZDI + SHVS[2015-2018]',
   'ZDi[2012-2015]',
   'ZXi[2015-2018]',
   'ZXi[2018-2022]',
   'VXi'],
  'S-Presso': ['VXi'],
  'Alto K10': ['VXi'],
  'Celerio': ['VXi', 'ZXi'],
  'Brezza': ['ZXi'],
  'S-Cross': ['Zeta 1.3[2014-2017]'],
  'Wagon R': ['VXI 1.0', 'LXI 1.0', 'LXI 1.0 CNG'],
  'Dzire': ['VXi']}}


selected_make = st.sidebar.selectbox("Select Make", options=list(d.keys()))

if selected_make:
    selected_model = st.sidebar.selectbox("Select Model", options=list(d[selected_make].keys()))
else:
    selected_model = None

# Dropdown for 'Variant' based on the selected 'Model'
if selected_model:
    selected_variant = st.sidebar.selectbox("Select Variant", options=d[selected_make][selected_model])
else:
    selected_variant = None



# makes = ["Maruti Suzuki"]
# selected_make = st.sidebar.selectbox("Make", makes)

# models = sorted(d.keys())
# selected_model = st.sidebar.selectbox("Model", models)

# variants = sorted(d[selected_model])
# selected_variant = st.sidebar.selectbox("Variant", variants)

selected_city = st.sidebar.selectbox(
    "City",
    [
        'Ahmedabad', 'Bangalore', 'Chennai', 'Gurgaon', 'Hyderabad', 'Kolkata',
        'Pune', 'Delhi', 'Panchkula', 'Ludhiana', 'Kharar', 'Coimbatore',
        'Noida', 'Ghaziabad', 'Lucknow', 'Mumbai', 'Thane', 'Mohali',
        'Kharagpur', 'Chandigarh', 'Ambala', 'Navi', 'Faridabad', 'Meerut',
        'Sangli', 'Surat', 'Mysore', 'Gulbarga', 'Ranga', 'Vadodara', 'Howrah'
    ]
)

selected_transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
selected_fuel_type = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])

# Numeric Inputs
age = st.sidebar.number_input("Age (years)", min_value=0, max_value=50, value=5, step=1)
distance = st.sidebar.number_input("Odometer Reading (km)", min_value=0, max_value=500000, value=40000, step=1000)

# Dynamic ± range slider
range_percentage = st.sidebar.slider("Confidence Range (%)", 1, 20, 5)

# -----------------------------------------------------------------------------
# 5) Prediction Function
# -----------------------------------------------------------------------------
def predict_price(age, distance, make, car_model, variant, city, transmission, fuel_type):
    """
    Get raw model prediction for the user input.
    """
    # Avoid division by zero
    distance_per_year = (distance / (age + 1)) if (age + 1) else distance

    input_data = pd.DataFrame([{
        'Make': make,
        'Model': car_model,
        'Transmission': transmission,
        'Fuel Type': fuel_type,
        'City': city,
        'Distance_numeric': distance,
        'Age': age,
        'Distance_per_year': np.round(distance_per_year, 2),
        'Variant': variant
    }])

    try:
        raw_pred = model.predict(input_data)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

    return raw_pred[0]

# -----------------------------------------------------------------------------
# 6) Main Button: Generate Prediction & Apply Guardrails
# -----------------------------------------------------------------------------
if st.button("Predict Price"):
    raw_price = predict_price(
        age=age,
        distance=distance,
        make=selected_make,
        car_model=selected_model,  # Renamed parameter to avoid shadowing
        variant=selected_variant,
        city=selected_city,
        transmission=selected_transmission,
        fuel_type=selected_fuel_type
    )

    if raw_price is None:
        st.error("An error occurred during prediction. Please check your inputs.")
    else:

        st.success(f"Predicted Price without guardrails: : ₹{round(raw_price)}")
        # Select subset based on nearest age group
        subset_mmv = get_nearest_age_subset(
            df=df,
            make=selected_make,
            model=selected_model,
            variant=selected_variant,
            age=age,
            min_samples=5,   # Minimum required samples
            max_delta=5      # Maximum age difference to consider
        )

        if subset_mmv.empty:
            st.error("No data available for the selected Make-Model-Variant.")
            st.success(f"Predicted Price: {raw_price}")

        else:
            # Apply guardrails with the M-M-V subset
            guarded_price = apply_guardrails(
                age=age,
                distance=distance,
                fuel_type=selected_fuel_type,
                city=selected_city,
                raw_prediction=raw_price,
                df_subset=subset_mmv,
                depreciation_rate=0.04,  # 2% depreciation rate per year beyond max age
                min_floor=40000           # ₹40,000 minimum floor
            )

            if guarded_price is None:
                st.error("Cannot predict a valid price for this vehicle under current regulations/constraints.")
            else:
                st.success(f"Predicted Price (Post-Guardrail): ₹{round(guarded_price)}")

                # Calculate dynamic ± range
                lower_bound = guarded_price * (1 - range_percentage / 100)
                upper_bound = guarded_price * (1 + range_percentage / 100)
                st.write(f"Price Range (±{range_percentage}%): ₹{round(lower_bound)} - ₹{round(upper_bound)}")

                # Show closest cars in the same M-M-V
                similar_cars = find_closest_cars(selected_make, selected_model, selected_variant, age, distance, df)
                if similar_cars.empty:
                    st.write("No similar cars found in the dataset for this M-M-V.")
                else:
                    st.write("Closest Cars (based on Age & Odometer):")
                    st.dataframe(similar_cars)

                # Optional: Plot Age vs. Price for M-M-V subset
                if not subset_mmv.empty:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(subset_mmv['Age'], subset_mmv['Price_numeric'],
                               color='blue', alpha=0.7, label='Dataset Cars (Same M-M-V)')
                    ax.scatter(age, guarded_price, color='red', s=100, zorder=5,marker = "*" , label='Predicted Car')

                    # Determine x-axis range for the plot
                    plot_max_age = max(subset_mmv['Age'].max(), age) + 5
                    plot_x = np.linspace(0, plot_max_age, 100)

                    # Shade the ± range
                    ax.fill_between(
                        x=plot_x,
                        y1=lower_bound,
                        y2=upper_bound,
                        color='lightgreen', alpha=0.2,
                        label=f"±{range_percentage}% Range"
                    )

                    ax.set_xlim(0, plot_max_age)
                    ax.set_xlabel("Age (Years)")
                    ax.set_ylabel("Price (₹)")
                    ax.set_title(f"Age vs. Price for {selected_make} {selected_model} {selected_variant}")
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.write("No data to plot for this M-M-V subset.")
