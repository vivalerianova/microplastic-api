from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import requests

app = FastAPI(title="Microplastic API (Render)")

class ForecastRequest(BaseModel):
    lat: float
    lon: float
    concentration: float
    days: int  # 1..5

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def km_to_deg_lat(km): 
    return km / 111.0

def km_to_deg_lon(km, lat):
    return km / (111.0 * np.cos(np.deg2rad(lat)) + 1e-9)

def fetch_currents(lat, lon, days):
    n = int(days) * 24
    try:
        url = "https://marine-api.open-meteo.com/v1/marine"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "ocean_current_velocity,ocean_current_direction",
            "forecast_days": int(clamp(days, 1, 5)),
            "timezone": "GMT",
        }
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        js = r.json()

        hourly = js.get("hourly", {}) or {}
        vel = hourly.get("ocean_current_velocity")
        direc = hourly.get("ocean_current_direction")
        if not vel or not direc:
            return np.zeros(n), np.zeros(n)

        vel = np.array(vel, dtype=float) / 3.6  # km/h -> m/s
        direc = np.deg2rad(np.array(direc, dtype=float))

        vel = vel[:n]
        direc = direc[:n]

        u = vel * np.sin(direc)  # east m/s
        v = vel * np.cos(direc)  # north m/s

        if len(u) < n:
            u = np.pad(u, (0, n - len(u)))
            v = np.pad(v, (0, n - len(v)))

        return u, v
    except Exception:
        return np.zeros(n), np.zeros(n)

def fetch_wind(lat, lon, days):
    n = int(days) * 24
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "wind_speed_10m,wind_direction_10m",
            "forecast_days": int(clamp(days, 1, 5)),
            "wind_speed_unit": "ms",
            "timezone": "GMT",
        }
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        js = r.json()

        hourly = js.get("hourly", {}) or {}
        spd = hourly.get("wind_speed_10m")
        direc = hourly.get("wind_direction_10m")
        if not spd or not direc:
            return np.zeros(n), np.zeros(n)

        spd = np.array(spd, dtype=float)
        direc = np.array(direc, dtype=float)

        spd = spd[:n]
        direc = direc[:n]

        toward = np.deg2rad((direc + 180.0) % 360.0)  # from->to
        u = spd * np.sin(toward)
        v = spd * np.cos(toward)

        if len(u) < n:
            u = np.pad(u, (0, n - len(u)))
            v = np.pad(v, (0, n - len(v)))

        return u, v
    except Exception:
        return np.zeros(n), np.zeros(n)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/forecast")
def forecast(req: ForecastRequest):
    lat = float(req.lat)
    lon = float(req.lon)
    days = int(clamp(req.days, 1, 5))
    conc = float(req.concentration)

    # частицы: как ты просила
    n_particles = int(clamp(conc, 1, 500))
    lats = np.full(n_particles, lat, dtype=float)
    lons = np.full(n_particles, lon, dtype=float)

    u_c, v_c = fetch_currents(lat, lon, days)
    u_w, v_w = fetch_wind(lat, lon, days)

    steps = min(len(u_c), len(u_w), days * 24)
    dt = 3600.0

    # лагранж: течение + 2% ветра
    for i in range(0, steps, 3):
        u = u_c[i] + 0.02 * u_w[i]
        v = v_c[i] + 0.02 * v_w[i]

        lats += (v * dt) / 111000.0
        lons += (u * dt) / (111000.0 * np.cos(np.deg2rad(lats)) + 1e-9)

        lats = np.clip(lats, -89.875, 89.875)
        lons = (lons + 180.0) % 360.0 - 180.0

    avg_speed_mps = float(np.mean(np.sqrt(u_c[:steps]**2 + v_c[:steps]**2))) if steps > 0 else 0.0

    # сетка 10×10 км вокруг точки (окно 100×100 км)
    cell_km = 10.0
    half_km = 50.0

    dlat = km_to_deg_lat(half_km)
    dlon = km_to_deg_lon(half_km, lat)

    min_lat = lat - dlat
    max_lat = lat + dlat
    min_lon = lon - dlon
    max_lon = lon + dlon

    step_lat = km_to_deg_lat(cell_km)
    step_lon = km_to_deg_lon(cell_km, lat)

    lat_edges = np.arange(min_lat, max_lat, step_lat)
    lon_edges = np.arange(min_lon, max_lon, step_lon)

    grid = []
    for la in lat_edges:
        for lo in lon_edges:
            count = np.sum(
                (lats >= la) & (lats < la + step_lat) &
                (lons >= lo) & (lons < lo + step_lon)
            )
            grid.append({
                "lat": float(la + step_lat / 2.0),
                "lon": float(lo + step_lon / 2.0),
                "value": float(count)
            })

    particles_preview = [
        {"lat": float(a), "lon": float(b)}
        for a, b in zip(lats[:500], lons[:500])
    ]

    return {
        "cell_km": int(cell_km),
        "days": days,
        "avg_speed_mps": avg_speed_mps,
        "grid": grid,
        "particles_preview": particles_preview
    }
