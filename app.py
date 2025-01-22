# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# def create_buckets(df):
#     df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 2, 5, 10, 20], labels=['0-2', '3-5', '6-10', '10+'], right=False)
#     return df


# def log1p_transform(x):
#     """Custom function used during pipeline training."""
#     import numpy as np
#     return np.log1p(x)


# # Load the model
# model = joblib.load('catboost_best_model.joblib')

# # Load the dataset (ensure you replace 'your_dataset.csv' with the actual path to your dataset)
# df = pd.read_csv('final_data_suzuki.csv')
# df = create_buckets(df)



# # Compute segment bounds
# segment_bounds = {}

# grouped = df.groupby(['Make', 'Model', 'Variant', 'AgeGroup'])
# for segment, group in grouped:
#     if len(group) > 5:  # Ensure sufficient data points per segment
#         p5 = np.percentile(group['Price_numeric'], 5)
#         p95 = np.percentile(group['Price_numeric'], 95)
#         segment_bounds[segment] = (p5, p95)

# # Define the clamping function
# def clamp_price(predicted_price, make, model, variant, age):
#     age_group = pd.cut([age], bins=[0, 2, 5, 10, 20], labels=['0-2', '3-5', '6-10', '10+'], right=False)[0]
#     segment = (make, model, variant, age_group)

#     if segment in segment_bounds:
#         p5, p95 = segment_bounds[segment]
#         if predicted_price < p5 * 0.95 or predicted_price > p95 * 1.05:
#             return (p5 + p95) / 2
#         else:
#             return predicted_price

#     # Check for alternative age groups
#     age_groups = ['0-2', '3-5', '6-10', '10+']
#     current_index = age_groups.index(age_group)

#     for i in range(len(age_groups)):
#         if i == current_index:
#             continue
#         alt_segment = (make, model, variant, age_groups[i])
#         if alt_segment in segment_bounds:
#             p5, p95 = segment_bounds[alt_segment]
#             if predicted_price < p5 * 0.95 or predicted_price > p95 * 1.05:
#                 return (p5 + p95) / 2
#             else:
#                 return predicted_price

#     return predicted_price


# # Streamlit application layout
# st.title("Used Car Price Prediction App")

# st.write("""
# This application predicts the price of a used car based on various features.
# Please fill in the details on the left sidebar and click 'Predict Price'.
# """)

# # Sidebar Inputs
# st.sidebar.header("Enter the Car Features")


# d = {'Alto 800': ['LXi', 'Lx[2012-2016]', 'VXi'],
#  'Baleno': ['Alpha 1.2[2015-2019]',
#   'Alpha[2019-2022]',
#   'Delta 1.2[2015-2019]',
#   'Zeta 1.2[2015-2019]',
#   'Zeta[2019-2022]'],
#  'Brezza': ['ZXi'],
#  'Celerio': ['VXi', 'ZXi'],
#  'Ciaz': ['Alpha 1.4 AT[2017-2018]',
#   'Alpha Hybrid 1.5 AT [2018-2020]',
#   'VXi[2014-2017]',
#   'ZXI+[2014-2017]',
#   'ZXi[2014-2017]'],
#  'Eeco': ['5 STR[2010-2022]'],
#  'Ertiga': ['VDI SHVS[2015-2018]',
#   'VXI[2015-2018]',
#   'VXi[2018-2022]',
#   'ZXi[2018-2022]'],
#  'Ignis': ['Delta 1.2 MT'],
#  'S-Cross': ['Zeta 1.3[2014-2017]'],
#  'S-Presso': ['VXi'],
#  'Swift DZire': ['VDI[2011-2015]', 'VXI[2011-2015]'],
#  'Swift': ['LXi', 'VDi[2014-2018]', 'VXi', 'VXi AMT', 'ZDi[2014-2018]', 'ZXi'],
#  'Vitara Brezza': ['VDi[2016-2020]', 'ZDi[2016-2020]'],
#  'Wagon R': ['LXI 1.0', 'LXI 1.0 CNG', 'VXI 1.0'],
#  'XL6': ['Zeta MT Petrol']}


# makes = ["Maruti Suzuki"]
# selected_make = st.sidebar.selectbox("Select Make", makes)

# # Get models (keys of the dictionary)
# models = sorted(d.keys())
# selected_model = st.sidebar.selectbox("Select Model", models)

# # Get variants for selected model
# variants = sorted(d[selected_model])
# selected_variant = st.sidebar.selectbox("Select Variant", variants)


# # Other Inputs
# selected_city = st.sidebar.selectbox(
#     "City",
#     [
#         'Ahmedabad', 'Bangalore', 'Chennai', 'Gurgaon', 'Hyderabad', 'Kolkata',
#         'Pune', 'Delhi', 'Panchkula', 'Ludhiana', 'Kharar', 'Coimbatore',
#         'Noida', 'Ghaziabad', 'Lucknow', 'Mumbai', 'Thane', 'Mohali',
#         'Kharagpur', 'Chandigarh', 'Ambala', 'Navi', 'Faridabad', 'Meerut',
#         'Sangli', 'Surat', 'Mysore', 'Gulbarga', 'Ranga', 'Vadodara', 'Howrah'
#     ]
# )

# selected_transmission = st.sidebar.selectbox(
#     "Transmission",
#     ['Manual', 'Automatic']
# )

# selected_fuel_type = st.sidebar.selectbox(
#     "Fuel Type",
#     ['Petrol', 'Diesel', 'CNG']
# )

# # Numeric inputs
# age = st.sidebar.number_input("Age (years)", min_value=0, max_value=50, value=5, step=1)
# distance = st.sidebar.number_input("Odometer Reading (km)", min_value=0, max_value=500000, value=40000, step=1000)

# # Function to make predictions
# def predict_price():
#     # Compute distance_per_year
#     distance_per_year = np.round(distance / (age + 1))
    
#     # Prepare input data
#     input_data = pd.DataFrame([{
#         'Model': selected_model,
#         'Transmission': selected_transmission,
#         'Fuel Type': selected_fuel_type,
#         'City': selected_city,
#         'Distance_numeric': distance,
#         'Age': age,
#         'Distance_per_year': distance_per_year,
#         'Variant': selected_variant
#     }])
    
#     # Predict using the model
#     prediction = model.predict(input_data)
#     return prediction[0]




# # Filter the dataset for price range
# filtered_data = df[
#     (df['Make'] == selected_make) &
#     (df['Model'] == selected_model) &
#     (df['Variant'] == selected_variant) 
# ]

# if not filtered_data.empty:
#     min_price = filtered_data['Price_numeric'].min()
#     max_price = filtered_data['Price_numeric'].max()
# else:
#     min_price = max_price = "N/A"


# def find_closest_cars():
#     # Filter dataset for the same MMV
#     filtered_data = df[
#         (df['Make'] == selected_make) &
#         (df['Model'] == selected_model) &
#         (df['Variant'] == selected_variant)
#     ]
    
#     # If no matching cars are found, return an empty DataFrame
#     if filtered_data.empty:
#         return pd.DataFrame()

#     # Compute the distance metric for each row
#     # Calculate differences
#     filtered_data['Age Difference'] = abs(filtered_data['Age'] - age)
#     filtered_data['Distance Difference'] = abs(filtered_data['Distance_numeric'] - distance)
    
#     # Sort by the distance metric
#     closest_cars = filtered_data.sort_values(by=['Age Difference', 'Distance Difference']).head(5)

#     return closest_cars[['Make', "Model" , "Variant" , "Age" , "Distance_numeric" , 'Price_numeric']]


# if st.button("Predict Price"):
#     predicted_price = predict_price()
#     st.success(f"Predicted Price: ₹{round(predicted_price)}")
#     clamped_price = clamp_price(predicted_price, selected_make, selected_model, selected_variant, age)
#     predicted_price = clamped_price
#     st.success(f"clamped Price: ₹{round(clamped_price)}")

#     lower_bound = predicted_price * 0.90
#     upper_bound = predicted_price * 1.10
#     st.write(f"Price Range (±5%): ₹{round(lower_bound)} - ₹{round(upper_bound)}")

#     closest_cars = find_closest_cars()

#     # Display closest cars
#     if closest_cars.empty:
#         st.write("No similar cars found in the dataset.")
#     else:
#         st.write("Closest Cars (based on age and odometer reading):")
#         st.dataframe(closest_cars)

#     filtered_data = df[
#         (df['Make'] == selected_make) &
#         (df['Model'] == selected_model) &
#         (df['Variant'] == selected_variant)
#     ]

#     if not filtered_data.empty:
#         # Scatterplot
#         fig, ax = plt.subplots(figsize=(8, 6))

#         # Plot all cars of the same MMV (blue dots)
#         ax.scatter(
#             filtered_data['Age'], 
#             filtered_data['Price_numeric'], 
#             color='blue', 
#             label='Cars of Same MMV', 
#             alpha=0.7
#         )

#         # Plot the predicted car (red dot)
#         ax.scatter(
#             age, 
#             predicted_price, 
#             color='red', 
#             label='Predicted Car', 
#             s=100, 
#             zorder=5
#         )

#         # Shade the ±5% price range
#         ax.fill_between(
#             x=[0, 15],
#             y1=lower_bound,
#             y2=upper_bound,
#             color='lightgreen',
#             alpha=0.3,
#             label='Predicted Price Range (±5%)'
#         )

#         # Add labels and title
#         ax.set_xlabel("Age (Years)")
#         ax.set_ylabel("Price (₹)")
#         ax.set_title(f"Age vs. Price for {selected_make} {selected_model} {selected_variant}")
#         ax.legend()

#         # Display the plot in Streamlit
#         st.pyplot(fig)
#     else:
#         st.write("No data available for the selected Make, Model, and Variant.")

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
    Assign age groups using carefully chosen bins, including 6-8, 8-10, and 10+ as requested.
    Also includes earlier buckets for completeness:
      0–2, 2–5, 5–6, 6–8, 8–10, 10+.
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

<<<<<<< HEAD
# -----------------------------------------------------------------------------
# 2) Load the model and dataset, then create buckets
# -----------------------------------------------------------------------------
model = joblib.load('catboost_best_model.joblib')
=======

# Load the model
model = joblib.load('best_model.joblib')

# Load the dataset (ensure you replace 'your_dataset.csv' with the actual path to your dataset)
>>>>>>> 67e016845b79e45bc616e584515c6ddcb5440689
df = pd.read_csv('final_data_suzuki.csv')
df = create_buckets(df)

# -----------------------------------------------------------------------------
# 3) Compute Segment Bounds (p5 and p95)
# -----------------------------------------------------------------------------
segment_bounds = {}
# Group by Make, Model, Variant, and AgeGroup
grouped = df.groupby(['Make', 'Model', 'Variant', 'AgeGroup'])
for segment, group in grouped:
    if len(group) > 5:  # Ensure sufficient data points per segment
        p5 = np.percentile(group['Price_numeric'], 5)
        p95 = np.percentile(group['Price_numeric'], 95)
        segment_bounds[segment] = (p5, p95)

# -----------------------------------------------------------------------------
# 4) Helper function to apply clamping based on segment bounds
# -----------------------------------------------------------------------------
def _apply_clamping(segment, predicted_price):
    """Helper: If predicted_price is outside [p5*0.95, p95*1.05], clamp to midpoint."""
    p5, p95 = segment_bounds[segment]
    lower_bound = p5 * 0.95
    upper_bound = p95 * 1.05
    if predicted_price < lower_bound or predicted_price > upper_bound:
        avg_price = (p5 + p95) / 2
        return avg_price
    else:
        return predicted_price


# -----------------------------------------------------------------------------
# 5) Clamping Function with Bucket Fallback and >10 Special Handling
# -----------------------------------------------------------------------------
def clamp_price(predicted_price, make, model, variant, age):
    """
    Clamps `predicted_price` based on segment bounds and the chosen age buckets.
    Also applies a year-based 10% discount beyond the last bucket boundary (age > 10),
    IF and ONLY IF the predicted price is out-of-range for that bucket.
    """
    # ------------------------------------------------------------
    # Determine the AgeGroup based on the new bins
    # ------------------------------------------------------------
    bins = [0, 2, 5, 6, 8, 10, float('inf')]
    labels = ['0-2', '2-5', '5-6', '6-8', '8-10', '10+']
    age_group = pd.cut([age], bins=bins, labels=labels, right=False)[0]
    
    # If for some reason age < 0, handle gracefully (unlikely scenario):
    if age_group is None:
        return predicted_price
    
    # ------------------------------------------------------------
    # Check if this is the last bucket (10+) or beyond its boundary
    # (In this example, last bucket starts at 10)
    # ------------------------------------------------------------
    last_bucket_start = bins[-2]  # = 10
    last_bucket_label = labels[-1]  # = '10+'
    
    # ------------------------------------------------------------
    # 1) Normal check: is there exact data for this AgeGroup?
    # ------------------------------------------------------------
    segment = (make, model, variant, age_group)
    if segment in segment_bounds:
        # We have data for this bucket
        clamped = _apply_clamping(segment, predicted_price)
        # If age actually exceeds the last bucket boundary (e.g., 12 years old),
        # but the predicted price fell inside the range -> No discount needed.
        # If out of range -> apply discount logic.
        if age > last_bucket_start:
            # The bucket is "10+"
            # Check if we actually had to clamp (meaning out of range)
            if clamped != predicted_price:
                # predicted_price was out of range -> apply discount
                # discount based on how many years beyond 'last_bucket_start'
                years_beyond = age - last_bucket_start
                discount_factor = max(1 - 0.10 * years_beyond, 0)
                
                # We discount from the *midpoint* of that last bucket
                p5, p95 = segment_bounds[segment]
                base_avg = (p5 + p95) / 2
                discounted_price = base_avg * discount_factor
                return discounted_price
            else:
                # It's in range, so return the clamped value (which might be original)
                return clamped
        else:
            # Age <= 10 or exactly 10
            return clamped
    
    # ------------------------------------------------------------
    # 2) If no exact bucket, try neighbor fallback (for <=10)
    # ------------------------------------------------------------
    # Only relevant if age <= 10, because if age > 10 we always
    # use the "10+" bucket.
    if age <= last_bucket_start:
        # valid_buckets excludes the final '10+' label if you want neighbor fallback
        valid_buckets = ['0-2', '2-5', '5-6', '6-8', '8-10']
        group_index = valid_buckets.index(age_group) if age_group in valid_buckets else None
        
        if group_index is not None:
            # Check immediate previous and next
            fallback_segments = []
            if group_index - 1 >= 0:
                fallback_segments.append((make, model, variant, valid_buckets[group_index - 1]))
            if group_index + 1 < len(valid_buckets):
                fallback_segments.append((make, model, variant, valid_buckets[group_index + 1]))
            
            for fb_segment in fallback_segments:
                if fb_segment in segment_bounds:
                    return _apply_clamping(fb_segment, predicted_price)
    
    # ------------------------------------------------------------
    # 3) If still nothing or age > 10 but no data found, 
    #    fallback to last-bucket discount logic if out of range
    # ------------------------------------------------------------
    #   i.e. if we do not have "10+" data at all, we cannot clamp properly.
    #   We'll only discount if user specifically wants to do so.  
    #   Otherwise, just return predicted_price as-is.
    
    # Check if age > last_bucket_start => apply discount if out-of-range
    if age > last_bucket_start:
        # Try to fallback to the last bucket label anyway
        last_segment = (make, model, variant, last_bucket_label)
        if last_segment in segment_bounds:
            # Check if predicted_price is out of range for that last bucket
            p5, p95 = segment_bounds[last_segment]
            lower_bound = p5 * 0.95
            upper_bound = p95 * 1.05
            if predicted_price < lower_bound or predicted_price > upper_bound:
                # predicted price is out of range => discount
                years_beyond = age - last_bucket_start
                discount_factor = max(1 - 0.05 * years_beyond, 0)
                base_avg = (p5 + p95) / 2
                discounted_price = base_avg * discount_factor
                return discounted_price
            else:
                return predicted_price
        else:
            return predicted_price
    else:
        # If no data for the current bucket or neighbors, just return
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
