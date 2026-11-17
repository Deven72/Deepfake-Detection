"""Heuristic risk bands for an experimental model score."""


def calculate_risk(deepfake_probability: float) -> dict:
    """Map model score to coarse bands; this is not calibrated confidence."""
    probability = float(deepfake_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be between 0.0 and 1.0")

    if probability < 0.30:
        level = "LOW"
    elif probability < 0.70:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return {
        "deepfake_probability": round(probability, 4),
        "risk_score": round(probability * 100, 2),
        "risk_level": level,
        "risk_interpretation": "Heuristic band from an experimental model score; not a verified probability of manipulation."
    }
