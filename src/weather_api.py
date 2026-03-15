import requests
from src.config import OPENWEATHER_API_KEY, OPENWEATHER_BASE_URL

def get_weather_by_city(city: str):
    """
    Returns temperature (°C) and humidity (%) from OpenWeather.
    Rainfall may not always be available.
    """
    if not OPENWEATHER_API_KEY:
        raise ValueError(
            "OpenWeather API key not found. Set OPENWEATHER_API_KEY in environment."
        )

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    res = requests.get(OPENWEATHER_BASE_URL, params=params, timeout=10)
    data = res.json()

    if res.status_code != 200:
        raise ValueError(f"Weather API error: {data}")

    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]

    # Rainfall (optional)
    rainfall = 0.0
    if "rain" in data:
        # OpenWeather may return 1h or 3h
        rainfall = data["rain"].get("1h", data["rain"].get("3h", 0.0))

    return {
        "temperature": float(temp),
        "humidity": float(humidity),
        "rainfall": float(rainfall)
    }
