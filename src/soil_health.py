def soil_health_score(n, p, k, ph):
    """
    Soil Health Score out of 100.
    Based on:
      - N, P, K balance
      - pH suitability
    """

    score = 100

    # Ideal ranges (general)
    if n < 50 or n > 120:
        score -= 15
    if p < 40 or p > 100:
        score -= 15
    if k < 40 or k > 100:
        score -= 15

    # pH check
    if ph < 5.5 or ph > 7.5:
        score -= 25
    elif 6.0 <= ph <= 7.0:
        score += 5

    # Clamp
    score = max(0, min(100, score))

    if score >= 85:
        level = "Excellent"
    elif score >= 70:
        level = "Good"
    elif score >= 50:
        level = "Average"
    else:
        level = "Poor"

    return score, level
