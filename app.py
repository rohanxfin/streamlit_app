import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def log1p_transform(x):
    """Custom function used during pipeline training."""
    import numpy as np
    return np.log1p(x)


# Load the model
model = joblib.load('best_model.joblib')

# Load the dataset (ensure you replace 'your_dataset.csv' with the actual path to your dataset)
df = pd.read_csv('final_data_suzuki.csv')

# Streamlit application layout
st.title("Used Car Price Prediction App")

st.write("""
This application predicts the price of a used car based on various features.
Please fill in the details on the left sidebar and click 'Predict Price'.
""")

# Sidebar Inputs
st.sidebar.header("Enter the Car Features")

# 1) Sequential dropdowns for Make, Model, and Variant
# Extract unique Makes
makes = sorted(df['Make'].unique())
selected_make = st.sidebar.selectbox("Select Make", makes)

# Filter models based on selected Make
models = sorted(df[df['Make'] == selected_make]['Model'].unique())
selected_model = st.sidebar.selectbox("Select Model", models)

# Filter variants based on selected Model
variants = sorted(df[(df['Make'] == selected_make) & (df['Model'] == selected_model)]['Variant'].unique())
selected_variant = st.sidebar.selectbox("Select Variant", variants)

# Other Inputs
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

selected_transmission = st.sidebar.selectbox(
    "Transmission",
    ['Manual', 'Automatic']
)

selected_fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    ['Petrol', 'Diesel', 'CNG']
)

# Numeric inputs
age = st.sidebar.number_input("Age (years)", min_value=0, max_value=50, value=5, step=1)
distance = st.sidebar.number_input("Odometer Reading (km)", min_value=0, max_value=500000, value=40000, step=1000)

# Function to make predictions
def predict_price():
    # Compute distance_per_year
    distance_per_year = np.round(distance / (age + 1))
    
    # Prepare input data
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
    
    # Predict using the model
    prediction = model.predict(input_data)
    return prediction[0]

# Filter the dataset for price range
filtered_data = df[
    (df['Make'] == selected_make) &
    (df['Model'] == selected_model) &
    (df['Variant'] == selected_variant) 
]

if not filtered_data.empty:
    min_price = filtered_data['Price_numeric'].min()
    max_price = filtered_data['Price_numeric'].max()
else:
    min_price = max_price = "N/A"


def find_closest_cars():
    # Filter dataset for the same MMV
    filtered_data = df[
        (df['Make'] == selected_make) &
        (df['Model'] == selected_model) &
        (df['Variant'] == selected_variant)
    ]
    
    # If no matching cars are found, return an empty DataFrame
    if filtered_data.empty:
        return pd.DataFrame()

    # Compute the distance metric for each row
    # Calculate differences
    filtered_data['Age Difference'] = abs(filtered_data['Age'] - age)
    filtered_data['Distance Difference'] = abs(filtered_data['Distance_numeric'] - distance)
    
    # Sort by the distance metric
    closest_cars = filtered_data.sort_values(by=['Age Difference', 'Distance Difference']).head(5)

    return closest_cars[['Make', "Model" , "Variant" , "Age" , "Distance_numeric" , 'Price_numeric']]

# Main Panel - Prediction
if st.button("Predict Price"):
    predicted_price = predict_price()
    lower_bound = predicted_price * 0.92
    upper_bound = predicted_price * 1.08

    st.success(f"Predicted Price: ₹{round(predicted_price)}")
    st.write(f"Price Range (±5%): ₹{round(lower_bound)} - ₹{round(upper_bound)}")


    closest_cars = find_closest_cars()
    
    # Display closest cars
    if closest_cars.empty:
        st.write("No similar cars found in the dataset.")
    else:
        st.write("Closest Cars (based on age and odometer reading):")
        st.dataframe(closest_cars)


    filtered_data = df[
        (df['Make'] == selected_make) &
        (df['Model'] == selected_model) &
        (df['Variant'] == selected_variant)
    ]

    if not filtered_data.empty:
        # Scatterplot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot all cars of the same MMV (blue dots)
        ax.scatter(
            filtered_data['Age'], 
            filtered_data['Price_numeric'], 
            color='blue', 
            label='Cars of Same MMV', 
            alpha=0.7
        )

        # Plot the predicted car (red dot)
        ax.scatter(
            age, 
            predicted_price, 
            color='red', 
            label='Predicted Car', 
            s=100, 
            zorder=5
        )

        # Add labels and title
        ax.set_xlabel("Age (Years)")
        ax.set_ylabel("Price (₹)")
        ax.set_title(f"Age vs. Price for {selected_make} {selected_model} {selected_variant}")
        ax.legend()

        # Display the plot in Streamlit
        st.pyplot(fig)
    else:
        st.write("No data available for the selected Make, Model, and Variant.")
