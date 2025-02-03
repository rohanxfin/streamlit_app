# utils.py
import numpy as np

def log1p_transform(x):
    """Custom log1p transform function used in the model's preprocessing pipeline."""
    return np.log1p(x)

def get_age_bracket(age):
    """Returns a label based on the age bucket."""
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
    """Returns a label based on the odometer bucket."""
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
    Starts with the same age, then expands the range by ±1, ±2, etc.
    """
    subset = df[(df['Make'] == make) & (df['Model'] == model) & (df['Variant'] == variant)]
    if subset.empty:
        return subset

    max_age = subset['Age'].max()
    if age > max_age:
        return subset

    delta = 0
    while delta <= max_delta:
        age_min = max(age - delta, 0)
        age_max = age + delta
        subset_age = subset[(subset['Age'] >= age_min) & (subset['Age'] <= age_max)]
        if len(subset_age) >= min_samples:
            return subset_age
        delta += 1

    return subset[(subset['Age'] >= max(age - max_delta, 0)) & (subset['Age'] <= min(age + max_delta, max_age))]

def find_closest_cars(make, model, variant, age, distance, df):
    """
    Returns up to 10 closest cars in the dataset based on the same M-M-V.
    """
    filtered = df[(df['Make'] == make) & (df['Model'] == model) & (df['Variant'] == variant)]
    if filtered.empty:
        return filtered

    filtered = filtered.copy()
    filtered['Age_Diff'] = abs(filtered['Age'] - age)
    filtered['Odom_Diff'] = abs(filtered['Distance_numeric'] - distance)
    closest_cars = filtered.sort_values(by=['Age_Diff', 'Odom_Diff']).head(10)
    return closest_cars[['Make', 'Model', 'Variant', 'Age', 'Distance_numeric', 'Price_numeric']]

# custom_transforms.py
import numpy as np

def log1p_transform(x):
    """Custom log1p transform function used in the model's preprocessing pipeline."""
    return np.log1p(x)