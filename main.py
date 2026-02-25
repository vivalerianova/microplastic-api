from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import requests
from global_land_mask import globe
import math

app = FastAPI()

class ForecastRequest(BaseModel):
    lat: float
    lon: float
    concentration: float
    days: int

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def km_to_deg_lat(km):
    return km / 111.0

def km_to_deg_lon(km, lat):
    return km / (111.0 * np.cos(np.deg2rad(lat)) + 1e-9)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/forecast")
def forecast(req: ForecastRequest):
    lat = req.lat
    lon = req.lon
    conc = req.concentration
    days = req.days

    n_particles = int(clamp(conc, 1, 500))

    # создаём частицы
    lats = np.random.normal(lat, 0.01, n_particles)
    lons = np.random.normal(lon, 0.01, n_particles)

    # простое смещение
    lats += 0.01 * days
    lons += 0.01 * days

    particles_preview = [
        {"lat": float(a), "lon": float(b)}
        for a, b in zip(lats, lons)
    ]

    # сетка 10 км
    cell_km = 10
    half_km = 50
    dlat = km_to_deg_lat(half_km)
    dlon = km_to_deg_lon(half_km, lat)

    min_lat = lat - dlat
    max_lat = lat + dlat
    min_lon = lon - dlon
    max_lon = lon + dlon

    step_lat = km_to_deg_lat(cell_km)
    step_lon = km_to_deg_lon(cell_km, lat)

    grid = []

    la = min_lat
    while la < max_lat:
        lo = min_lon
        while lo < max_lon:
            count = np.sum(
                (lats >= la) & (lats < la + step_lat) &
                (lons >= lo) & (lons < lo + step_lon)
            )
            grid.append({
                "lat": float(la + step_lat/2),
                "lon": float(lo + step_lon/2),
                "value": float(count)
            })
            lo += step_lon
        la += step_lat

    avg_speed_mps = 0.15

    return {
        "cell_km": cell_km,
        "days": days,
        "avg_speed_mps": avg_speed_mps,
        "grid": grid,
        "particles_preview": particles_preview
    }
