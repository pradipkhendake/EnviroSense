from flask import Flask, render_template, request
import requests
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

cities = {
    "Pune": (18.5204, 73.8567),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707)
}

def get_air_quality_message(pm25_list):
    if not pm25_list:
        return "Air quality data is not available."

    avg_pm25 = sum(pm25_list) / len(pm25_list)

    if avg_pm25 <= 12:
        return "🌿 Excellent Air Quality – Breathe Freely!"
    elif avg_pm25 <= 35.4:
        return "🙂 Good Air Quality – Minimal impact on health."
    elif avg_pm25 <= 55.4:
        return "⚠️ Moderate Air Quality – Unhealthy for sensitive groups."
    elif avg_pm25 <= 150.4:
        return "🚨 Poor Air Quality – Limit outdoor activity."
    else:
        return "☠️ Hazardous Air Quality – Stay Indoors!"

def predict_pm25(pm25_values):
    clean = [(i, v) for i, v in enumerate(pm25_values) if isinstance(v, (int, float, float))]
    if len(clean) < 6:
        return [], []

    X = np.array([i for i, _ in clean]).reshape(-1, 1)
    y = np.array([v for _, v in clean])
    model = LinearRegression().fit(X, y)

    future_X = np.array([len(clean) + i for i in range(1, 7)]).reshape(-1, 1)
    future_y = model.predict(future_X)

    return [f"T+{i}h" for i in range(1, 7)], future_y.tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    selected_city = request.form.get("city", "Pune")
    lat, lon = cities[selected_city]

    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    air_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm10,pm2_5,carbon_monoxide&timezone=auto"

    try:
        weather_res = requests.get(weather_url).json()
        weather = weather_res.get("current_weather", {})
        temperature = weather.get("temperature", "N/A")
        windspeed = weather.get("windspeed", "N/A")
        time = weather.get("time", "N/A")

        air_res = requests.get(air_url).json()
        air = air_res.get("hourly", {})

        pm10 = air.get("pm10", [])[0] if air.get("pm10") else "N/A"
        pm25 = air.get("pm2_5", [])[0] if air.get("pm2_5") else "N/A"
        co = air.get("carbon_monoxide", [])[0] if air.get("carbon_monoxide") else "N/A"

        times = air.get("time", [])[:24]
        pm25_values = air.get("pm2_5", [])[:24]
        pm25_clean = [v for v in pm25_values if isinstance(v, (int, float))]
        pm25_message = get_air_quality_message(pm25_clean)

        # Prediction
        pred_labels, pred_values = predict_pm25(pm25_values)

        return render_template("index.html",
                               cities=cities.keys(),
                               selected_city=selected_city,
                               temp=temperature,
                               wind=windspeed,
                               time=time,
                               pm10=pm10,
                               pm25=pm25,
                               co=co,
                               chart_labels=times,
                               chart_values=pm25_values,
                               pred_labels=pred_labels,
                               pred_values=pred_values,
                               pm25_message=pm25_message)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
