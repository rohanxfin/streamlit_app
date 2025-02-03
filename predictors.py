# predictors.py
import joblib
import numpy as np
import pandas as pd
from config import MODEL_PATH_CAT, MODEL_PATH_CAT2, MODEL_PATH_XGB, MODEL_PATH_DT

def predict_price_from_multiple_models(age, distance, make, car_model, variant, city, transmission, fuel_type):
    """
    Get raw model predictions from multiple models.
    """
    # Avoid division by zero
    distance_per_year = (distance / (age + 1))
    
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
    
    predictions = {}
    models = {
        "CatBoost": MODEL_PATH_CAT,
        "Cat2": MODEL_PATH_CAT2,
        "XGBoost": MODEL_PATH_XGB,
        "Decision Tree": MODEL_PATH_DT,
    }

    for model_name, model_path in models.items():
        try:
            # Debug: Log the model being loaded
            print(f"Attempting to load model: {model_name} from {model_path}")
            model = joblib.load(model_path)
            prediction = model.predict(input_data)
            predictions[model_name] = prediction[0]
            # Debug: Print the prediction for this model
            print(f"{model_name} prediction: {prediction[0]}")
        except Exception as e:
            # Log the error and assign None for this model
            print(f"Error during prediction with {model_name}: {e}")
            predictions[model_name] = None
    return predictions

