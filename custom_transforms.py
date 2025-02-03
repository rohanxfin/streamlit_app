# custom_transforms.py
import numpy as np

def log1p_transform(x):
    """Custom log1p transform function used in the model's preprocessing pipeline."""
    return np.log1p(x)
