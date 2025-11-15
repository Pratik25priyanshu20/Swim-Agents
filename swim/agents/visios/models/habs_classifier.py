# swim/agents/visios/models/hab_classifier.py

import random

def classify_hab_from_image(image_path: str) -> dict:
    """
    Simulates classification of a lake image for HAB detection.
    
    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: Prediction results with probabilities and label.
    """
    # Simulated output
    prediction = {
        "image": image_path,
        "bloom_probability": round(random.uniform(0.1, 0.95), 2),
        "cyanobacteria_density": round(random.uniform(50, 1200), 1),
        "toxin_levels": round(random.uniform(0, 5), 2),
    }

    # Add human-readable interpretation
    prob = prediction["bloom_probability"]
    if prob > 0.75:
        prediction["bloom_risk"] = "High"
    elif prob > 0.45:
        prediction["bloom_risk"] = "Moderate"
    else:
        prediction["bloom_risk"] = "Low"

    return prediction