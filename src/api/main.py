from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .data_fetcher import get_features_for_point, get_tensor_for_point, get_spread_tensor_for_point
from .inference import predict_ignition, predict_spread

app = FastAPI(title="IberFire API", description="API para predicción de ignición y propagación de incendios en España.")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IgnitionRequest(BaseModel):
    lat: float
    lon: float
    date: str

class SpreadRequest(BaseModel):
    lat: float
    lon: float
    date: str

@app.get("/")
def read_root():
    return {"message": "Welcome to IberFire API"}

@app.post("/predict/ignition")
async def ignition(req: IgnitionRequest):
    print(f"\n[API REQUEST] Modelo Ignición -> Fecha: {req.date} | Lat: {req.lat:.4f} | Lon: {req.lon:.4f}")
    
    # Validation 1: Geographical Box (Iberian Peninsula)
    if not (35.0 <= req.lat <= 44.0) or not (-10.0 <= req.lon <= 5.0):
        return {"error": "Las coordenadas ingresadas están fuera de la Península Ibérica."}
        
    # Validation 2: Temporal Box (2008 to Today)
    from datetime import datetime
    try:
        req_date = datetime.strptime(req.date, "%Y-%m-%d").date()
        today = datetime.today().date()
        min_date = datetime.strptime("2008-01-01", "%Y-%m-%d").date()
        
        if req_date < min_date:
            return {"error": "No hay datos climáticos ni topográficos disponibles antes del 2008."}
        if req_date > today:
            return {"error": "No se pueden predecir igniciones en fechas futuras no modeladas."}
    except ValueError:
        return {"error": "Formato de fecha inválido. Debe ser YYYY-MM-DD."}

    # 1. Fetch features and build tensor
    tensor_np, features = await get_tensor_for_point(req.lat, req.lon, req.date)
    if "error" in features:
        return features
        
    # 2. Predict ignition risk
    result = await predict_ignition(tensor_np, features)
    
    # 3. Bundle metadata for frontend
    final_response = {
        **result,
        "cell_bounds": features.get("cell_bounds", []),
        "features": features.get("features", {}),
        "x_3035": features.get("x_3035"),
        "y_3035": features.get("y_3035"),
    }

    import json
    print("\n[API RESPONSE] --- Datos Enviados al Frontend (Ignición) ---")
    # Log everything except the large polygon data if it exists
    print(json.dumps({k:v for k,v in final_response.items() if k != "geojson_polygon"}, indent=2))
    print("----------------------------------------------------------\n")
    
    return final_response

@app.post("/predict/spread")
async def spread(req: SpreadRequest):
    print(f"\n[API REQUEST] Modelo Propagación -> Fecha: {req.date} | Lat: {req.lat:.4f} | Lon: {req.lon:.4f}")
    
    # Validation 1: Geographical Box (Iberian Peninsula)
    if not (35.0 <= req.lat <= 44.0) or not (-10.0 <= req.lon <= 5.0):
        return {"error": "Las coordenadas ingresadas están fuera de la Península Ibérica."}
        
    # 1. Fetch features and build tensor
    tensor_np, features = await get_spread_tensor_for_point(req.lat, req.lon, req.date)
    if "error" in features:
        return features
        
    # 2. Predict spread polygon
    result = await predict_spread(tensor_np, features)
    
    final_response = {
        **result,
        "cell_bounds": features.get("cell_bounds", []),
        "features": features.get("features", {}),
        "x_3035": features.get("x_3035"),
        "y_3035": features.get("y_3035"),
    }

    import json
    print("\n[API RESPONSE] --- Datos Enviados al Frontend (Propagación) ---")
    print(json.dumps({k:v for k,v in final_response.items() if k != "geojson_polygon"}, indent=2))
    print("-------------------------------------------------------------\n")

    return final_response
