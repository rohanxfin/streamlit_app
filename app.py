import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1) Create Logical Age Buckets
# -----------------------------------------------------------------------------
def create_buckets(df):
    """
    Assigns age groups using these bins:
      0-2, 2-5, 5-6, 6-8, 8-10, 10+.
    Adjust as needed, but be sure the bins and labels match in length.
    """
    bins = [0, 2, 5, 6, 8, 10, float('inf')]
    labels = ['0-2', '2-5', '5-6', '6-8', '8-10', '10+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    return df


# -----------------------------------------------------------------------------
# Custom log1p transform (if needed in the model pipeline)
# -----------------------------------------------------------------------------
def log1p_transform(x):
    import numpy as np
    return np.log1p(x)

# -----------------------------------------------------------------------------
# 2) Load the model and dataset, then create buckets
# -----------------------------------------------------------------------------
model = joblib.load('catboost_best_model.joblib')

# Load the model
model = joblib.load('best_model.joblib')

# Load the dataset (ensure you replace 'your_dataset.csv' with the actual path to your dataset)
df = pd.read_csv('final_data_suzuki.csv')
df = create_buckets(df)

# -----------------------------------------------------------------------------
# 3) Compute Segment Bounds (p5 and p95)
# -----------------------------------------------------------------------------
segment_bounds = {}   # (make, model, variant, bucket) -> (p5, p95)

df = create_buckets(df)  # apply the function

grouped = df.groupby(['Make', 'Model', 'Variant', 'AgeGroup'])
for segment, group in grouped:
    if len(group) > 5:  # ensure enough data
        p5 = np.percentile(group['Price_numeric'], 5)
        p95 = np.percentile(group['Price_numeric'], 95)
        segment_bounds[segment] = (p5, p95)


import numpy as np

segment_bounds = {}   # (make, model, variant, bucket) -> (p5, p95)

df = create_buckets(df)  # apply the function

grouped = df.groupby(['Make', 'Model', 'Variant', 'AgeGroup'])
for segment, group in grouped:
    if len(group) > 5:  # ensure enough data
        p5 = np.percentile(group['Price_numeric'], 5)
        p95 = np.percentile(group['Price_numeric'], 95)
        segment_bounds[segment] = (p5, p95)


all_labels = ['0-2', '2-5', '5-6', '6-8', '8-10', '10+']
all_bins =    [  0,    2,    5,    6,    8,   10,    float('inf') ]

def get_last_available_bucket(make, model, variant):
    """
    Returns (last_label, last_boundary_start) for the highest bucket
    that actually has data in segment_bounds for this M/M/V.
    If no data at all, returns (None, None).
    """
    # Gather all labels that exist for this M/M/V
    existing_labels = []
    for lbl in all_labels:
        seg = (make, model, variant, lbl)
        if seg in segment_bounds:
            existing_labels.append(lbl)
    
    if not existing_labels:
        return None, None  # no data at all

    # Among existing labels, pick the one with the highest index in all_labels
    last_label = max(existing_labels, key=lambda x: all_labels.index(x))
    
    # Example: if last_label == '8-10', its index is 4 (0-based in the all_labels list)
    idx = all_labels.index(last_label)
    last_bucket_start = all_bins[idx]  # e.g., if label is '8-10', start is 8
    return last_label, last_bucket_start




# -----------------------------------------------------------------------------
# 4) Helper function to apply clamping based on segment bounds
# -----------------------------------------------------------------------------
def _apply_clamping(segment, predicted):
    """
    Check if 'predicted' is within [p5*0.95, p95*1.05].
    If outside, return midpoint; else return predicted.
    """
    p5, p95 = segment_bounds[segment]
    lb, ub = p5 * 0.95, p95 * 1.05
    if predicted < lb or predicted > ub:
        return (p5 + p95) / 2
    return predicted


def clamp_price(predicted_price, make, model, variant, age):
    """
    1) Find age_group from the bins (0-2, 2-5, 5-6, 6-8, 8-10, 10+).
    2) If we have data for that bucket, clamp if out of range.
    3) If not, fallback to neighbors (if age <= 'last' boundary).
    4) If age is beyond the last available bucket boundary and
       predicted price is out-of-range for that last bucket, 
       discount per extra year (10% each year).
    """
    # ----------------------------------------------------
    # Identify the age_group from the bins
    # ----------------------------------------------------
    bins = [0, 2, 5, 6, 8, 10, float('inf')]
    labels = ['0-2', '2-5', '5-6', '6-8', '8-10', '10+']
    age_group = pd.cut([age], bins=bins, labels=labels, right=False)[0]
    if age_group is None:
        # In case of negative or weird age
        return predicted_price

    # ----------------------------------------------------
    # Get the last available bucket for this M/M/V
    # ----------------------------------------------------
    last_label, last_boundary = get_last_available_bucket(make, model, variant)
    # If no data at all for M/M/V, we cannot clamp logically
    if last_label is None:
        return predicted_price  # no data => return as is

    # ----------------------------------------------------
    # 1) Attempt normal clamp in the exact age_group
    # ----------------------------------------------------
    seg = (make, model, variant, age_group)
    if seg in segment_bounds:
        clamped_val = _apply_clamping(seg, predicted_price)
        # If age > last_boundary AND out-of-range → discount logic
        # but we only discount if we truly had to clamp 
        # (meaning original price was out of range).
        was_clamped = (clamped_val != predicted_price)
        if age > last_boundary and was_clamped:
            # apply discount from the midpoint of the last bucket (not necessarily '10+')
            last_seg = (make, model, variant, last_label)
            p5, p95 = segment_bounds[last_seg]
            base_avg = (p5 + p95) / 2

            years_beyond = age - last_boundary
            discount_factor = max(1 - 0.10 * years_beyond, 0)
            discounted = base_avg * discount_factor
            return discounted
        else:
            return clamped_val
    # ----------------------------------------------------
    # 2) If we have no exact bucket for the user's age_group
    #    we can do neighbor fallback if age <= last_boundary
    # ----------------------------------------------------
    all_labels = ['0-2', '2-5', '5-6', '6-8', '8-10', '10+']
    idx = all_labels.index(age_group)
    
    # Only do fallback if the age is not beyond the last boundary
    # (because if it is, we want to rely on the last bucket discount logic).
    if age <= last_boundary:
        # Check neighbors in ascending order of distance 
        fallback_candidates = []
        if idx - 1 >= 0:
            fallback_candidates.append((make, model, variant, all_labels[idx - 1]))
        if idx + 1 < len(all_labels):
            fallback_candidates.append((make, model, variant, all_labels[idx + 1]))
        
        for fb_seg in fallback_candidates:
            if fb_seg in segment_bounds:
                fb_clamped = _apply_clamping(fb_seg, predicted_price)
                return fb_clamped
    
    # ----------------------------------------------------
    # 3) If still nothing, or the age is beyond last boundary:
    #    Use the "last available bucket" logic. 
    # ----------------------------------------------------
    # We'll see if the predicted price is out-of-range for that last bucket:
    last_seg = (make, model, variant, last_label)
    if last_seg in segment_bounds:
        # Check if out-of-range => discount
        p5, p95 = segment_bounds[last_seg]
        lb, ub = p5 * 0.95, p95 * 1.05
        if predicted_price < lb or predicted_price > ub:
            # out of range => discount from midpoint
            base_avg = (p5 + p95) / 2
            # how many years beyond that boundary?
            if age > last_boundary:
                years_beyond = age - last_boundary
            else:
                years_beyond = 0  # e.g. if last boundary is 8 but age is 7
            discount_factor = max(1 - 0.10 * years_beyond, 0)
            return base_avg * discount_factor
        else:
            return predicted_price
    else:
        # truly no data for last_label => we can't clamp or discount
        return predicted_price


# -----------------------------------------------------------------------------
# 6) Streamlit Application Layout and Inputs
# -----------------------------------------------------------------------------
st.title("Used Car Price Prediction App")
st.write("""
This application predicts the price of a used car based on various features.
Please fill in the details on the left sidebar and click 'Predict Price'.
""")

# Sidebar Inputs
st.sidebar.header("Enter the Car Features")

d = {
    'Alto 800': ['LXi', 'Lx[2012-2016]', 'VXi'],
    'Baleno': ['Alpha 1.2[2015-2019]', 'Alpha[2019-2022]', 'Delta 1.2[2015-2019]', 'Zeta 1.2[2015-2019]', 'Zeta[2019-2022]'],
    'Brezza': ['ZXi'],
    'Celerio': ['VXi', 'ZXi'],
    'Ciaz': ['Alpha 1.4 AT[2017-2018]', 'Alpha Hybrid 1.5 AT [2018-2020]', 'VXi[2014-2017]', 'ZXI+[2014-2017]', 'ZXi[2014-2017]'],
    'Eeco': ['5 STR[2010-2022]'],
    'Ertiga': ['VDI SHVS[2015-2018]', 'VXI[2015-2018]', 'VXi[2018-2022]', 'ZXi[2018-2022]'],
    'Ignis': ['Delta 1.2 MT'],
    'S-Cross': ['Zeta 1.3[2014-2017]'],
    'S-Presso': ['VXi'],
    'Swift DZire': ['VDI[2011-2015]', 'VXI[2011-2015]'],
    'Swift': ['LXi', 'VDi[2014-2018]', 'VXi', 'VXi AMT', 'ZDi[2014-2018]', 'ZXi'],
    'Vitara Brezza': ['VDi[2016-2020]', 'ZDi[2016-2020]'],
    'Wagon R': ['LXI 1.0', 'LXI 1.0 CNG', 'VXI 1.0'],
    'XL6': ['Zeta MT Petrol']
}

makes = ["Maruti Suzuki"]
selected_make = st.sidebar.selectbox("Select Make", makes)

# Get models (keys of the dictionary)
models = sorted(d.keys())
selected_model = st.sidebar.selectbox("Select Model", models)

# Get variants for the selected model
variants = sorted(d[selected_model])
selected_variant = st.sidebar.selectbox("Select Variant", variants)

# Other Inputs
selected_city = st.sidebar.selectbox(
    "City",
    ['Ahmedabad', 'Bangalore', 'Chennai', 'Gurgaon', 'Hyderabad', 'Kolkata',
     'Pune', 'Delhi', 'Panchkula', 'Ludhiana', 'Kharar', 'Coimbatore',
     'Noida', 'Ghaziabad', 'Lucknow', 'Mumbai', 'Thane', 'Mohali',
     'Kharagpur', 'Chandigarh', 'Ambala', 'Navi', 'Faridabad', 'Meerut',
     'Sangli', 'Surat', 'Mysore', 'Gulbarga', 'Ranga', 'Vadodara', 'Howrah']
)

selected_transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
selected_fuel_type = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])

# Numeric Inputs
age = st.sidebar.number_input("Age (years)", min_value=0, max_value=50, value=5, step=1)
distance = st.sidebar.number_input("Odometer Reading (km)", min_value=0, max_value=500000, value=40000, step=1000)

# -----------------------------------------------------------------------------
# 7) Prediction Functions
# -----------------------------------------------------------------------------
def predict_price():
    # Compute distance_per_year
    distance_per_year = np.round(distance / (age + 1))
    
    # Prepare input data DataFrame
    input_data = pd.DataFrame([{
        'Model': selected_model,
        'Transmission': selected_transmission,
        'Fuel Type': selected_fuel_type,
        'City': selected_city,
        'Distance_numeric': distance,
        'Age': age,
        'Distance_per_year': distance_per_year,
        'Variant': selected_variant
    }])
    
    prediction = model.predict(input_data)
    return prediction[0]

def find_closest_cars():
    # Filter dataset for the same Make, Model, Variant
    filtered_data = df[
        (df['Make'] == selected_make) &
        (df['Model'] == selected_model) &
        (df['Variant'] == selected_variant)
    ]
    
    if filtered_data.empty:
        return pd.DataFrame()
    
    filtered_data['Age Difference'] = abs(filtered_data['Age'] - age)
    filtered_data['Distance Difference'] = abs(filtered_data['Distance_numeric'] - distance)
    closest_cars = filtered_data.sort_values(by=['Age Difference', 'Distance Difference']).head(5)
    return closest_cars[['Make', 'Model', 'Variant', 'Age', 'Distance_numeric', 'Price_numeric']]

# -----------------------------------------------------------------------------
# 8) Run Prediction on Button Click & Display Results
# -----------------------------------------------------------------------------
if st.button("Predict Price"):
    predicted_price = predict_price()
    st.success(f"Predicted Price: ₹{round(predicted_price)}")
    
    clamped_price = clamp_price(predicted_price, selected_make, selected_model, selected_variant, age)
    st.success(f"Clamped Price: ₹{round(clamped_price)}")
    
    lower_bound = clamped_price * 0.95
    upper_bound = clamped_price * 1.15
    st.write(f"Price Range (±5%): ₹{round(lower_bound)} - ₹{round(upper_bound)}")
    
    closest_cars = find_closest_cars()
    if closest_cars.empty:
        st.write("No similar cars found in the dataset.")
    else:
        st.write("Closest Cars (based on age and odometer reading):")
        st.dataframe(closest_cars)
    
    # Plotting
    filtered_data = df[
        (df['Make'] == selected_make) &
        (df['Model'] == selected_model) &
        (df['Variant'] == selected_variant)
    ]
    
    if not filtered_data.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(filtered_data['Age'], filtered_data['Price_numeric'],
                   color='blue', label='Cars of Same MMV', alpha=0.7)
        ax.scatter(age, clamped_price, color='red', label='Predicted Car', s=100, zorder=5)
        ax.fill_between(x=[0, 15], y1=lower_bound, y2=upper_bound,
                        color='lightgreen', alpha=0.3, label='Predicted Price Range (±5%)')
        ax.set_xlabel("Age (Years)")
        ax.set_ylabel("Price (₹)")
        ax.set_title(f"Age vs. Price for {selected_make} {selected_model} {selected_variant}")
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("No data available for the selected Make, Model, and Variant.")
