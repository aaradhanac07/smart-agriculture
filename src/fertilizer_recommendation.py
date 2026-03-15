def fertilizer_advice(n, p, k):
    """
    Simple nutrient advice based on typical ranges.
    You can adjust thresholds later.
    """
    advice = []

    # Nitrogen
    if n < 50:
        advice.append("Nitrogen is LOW → add Urea / Nitrogen fertilizer.")
    elif n > 120:
        advice.append("Nitrogen is HIGH → reduce nitrogen fertilizer.")
    else:
        advice.append("Nitrogen is OPTIMAL.")

    # Phosphorus
    if p < 40:
        advice.append("Phosphorus is LOW → add DAP / SSP fertilizer.")
    elif p > 100:
        advice.append("Phosphorus is HIGH → reduce phosphorus fertilizer.")
    else:
        advice.append("Phosphorus is OPTIMAL.")

    # Potassium
    if k < 40:
        advice.append("Potassium is LOW → add MOP / Potash fertilizer.")
    elif k > 100:
        advice.append("Potassium is HIGH → reduce potash fertilizer.")
    else:
        advice.append("Potassium is OPTIMAL.")

    return advice
