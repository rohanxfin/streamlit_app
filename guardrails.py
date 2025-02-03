# guardrails.py
import numpy as np
from utils import get_age_bracket

def apply_guardrails(age, distance, fuel_type, city, avg_prediction, df_subset,
                     depreciation_rate=0.02, min_floor=50000, appreciation_rate=0.05):
    """
    Applies guardrails on the average prediction and returns a final price.
    """
    # Regulatory constraints for certain regions and fuel types
    if (fuel_type.lower() == 'diesel' and age > 10 and city.lower() in ['delhi', 'gurgaon', 'noida']):
        return None
    if (fuel_type.lower() == 'petrol' and age > 15 and city.lower() in ['delhi', 'gurgaon', 'noida']):
        return None

    df_subset = df_subset.copy()
    df_subset['Age_Bracket'] = df_subset['Age'].apply(get_age_bracket)
    min_age_subset = df_subset['Age'].min()
    max_age_subset = df_subset['Age'].max()

    clamped_price = avg_prediction

    # For cars newer than those in the dataset
    if age < min_age_subset:
        input_bracket = get_age_bracket(age)
        bracket_order = ['0-3', '3-5', '5-10', '10-15', '15+']
        next_bracket = None
        for bracket in bracket_order[bracket_order.index(input_bracket)+1:]:
            if bracket in df_subset['Age_Bracket'].unique():
                next_bracket = bracket
                break
        if next_bracket:
            next_subset = df_subset[df_subset['Age_Bracket'] == next_bracket]
            if not next_subset.empty:
                base_price = next_subset['Price_numeric'].quantile(0.75) if len(next_subset) >= 5 \
                             else next_subset['Price_numeric'].mean()
                years_below = min_age_subset - age
                clamped_price = base_price * ((1 + appreciation_rate) ** years_below)
        else:
            clamped_price = avg_prediction

    # For cars within the dataset age range
    elif age <= max_age_subset and len(df_subset) >= 1:
        if len(df_subset) >= 5:
            lower_p = 25 if len(df_subset) < 20 else 5
            upper_p = 75 if len(df_subset) < 20 else 95
            p_low, p_high = np.percentile(df_subset['Price_numeric'], [lower_p, upper_p])
            clamped_price = np.clip(avg_prediction, p_low * 0.95, p_high * 1.05)
        else:
            clamped_price = df_subset['Price_numeric'].mean()

    # For cars older than those in the dataset
    if age > max_age_subset and len(df_subset) > 0:
        last_bracket_subset = df_subset[df_subset['Age'] == max_age_subset]
        if not last_bracket_subset.empty:
            base_price = last_bracket_subset['Price_numeric'].quantile(0.25) if len(last_bracket_subset) >= 5 \
                         else last_bracket_subset['Price_numeric'].mean()
            if fuel_type.lower() == 'diesel' and city.lower() in ['delhi', 'gurgaon', 'noida']:
                depreciation_rate = 0.07
            elif fuel_type.lower() == 'petrol' and age > 15:
                depreciation_rate = 0.05
            years_beyond = age - max_age_subset
            clamped_price = base_price * ((1 - depreciation_rate) ** years_beyond)
        else:
            clamped_price = avg_prediction

    final_price = max(clamped_price, min_floor)
    return final_price
