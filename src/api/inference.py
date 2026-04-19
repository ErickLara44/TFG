import torch
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import MODEL_SAVE_PATH

# Try to load models lazily to avoid delay at startup if not used
__ignition_model = None
__spread_model = None

def get_ignition_model():
    global __ignition_model
    if __ignition_model is None:
        try:
            if MODEL_SAVE_PATH.exists():
                print(f"Loading Ignition model from {MODEL_SAVE_PATH}")
                from src.models.ignition import RobustFireIgnitionModel
                
                # Instantiate model (using the known parameters from training)
                model = RobustFireIgnitionModel(
                    num_input_channels=18, 
                    temporal_context=3, 
                    hidden_dims=[64, 128], 
                    dropout=0.2
                )
                
                checkpoint = torch.load(MODEL_SAVE_PATH, map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif isinstance(checkpoint, dict) and not any(k.startswith('multi_scale_lstm') for k in checkpoint.keys()):
                    # Could be a different format, but let's assume it has layers if it has 'multi_scale_lstm'
                    pass
                else:
                    # Direct state dict
                    model.load_state_dict(checkpoint)
                
                model.eval()
                __ignition_model = model
            else:
                print(f"Warning: Model not found at {MODEL_SAVE_PATH}. Using dummy model.")
                __ignition_model = "DUMMY"
        except Exception as e:
            print(f"Error loading model: {e}")
            __ignition_model = "DUMMY"
    return __ignition_model

async def predict_ignition(tensor_np: np.ndarray, features: dict) -> dict:
    """
    Runs the ignition model inference on the provided features and spatial tensor.
    """
    if "error" in features:
        return {"error": features["error"]}

    model = get_ignition_model()
    
    prob = 0.0
    risk = "Unknown"
    
    if hasattr(model, "forward"):
        try:
            device = next(model.parameters()).device
            input_tensor = torch.from_numpy(tensor_np).to(device)
            
            # --- DEBUG LOGGING ---
            max_val = input_tensor.max().item()
            min_val = input_tensor.min().item()
            mean_val = input_tensor.mean().item()
            
            # Identify channel with max value
            # input_tensor is (1, 3, 18, 64, 64)
            # Find max over batch(0), time(1), y(3), x(4) to leave channel(2)
            channel_maxes = input_tensor.max(dim=0)[0].max(dim=0)[0].max(dim=1)[0].max(dim=1)[0]
            max_idx = torch.argmax(channel_maxes).item()
            
            var_names = [
                'elevation_mean', 'slope_mean',
                'CLC_2018_forest_proportion', 'CLC_2018_scrub_proportion', 'CLC_2018_agricultural_proportion',
                'dist_to_roads_mean', 'popdens_2018', 'is_waterbody',
                't2m_mean', 'RH_min', 'wind_speed_mean', 'total_precipitation_mean',
                'NDVI', 'SWI_010', 'FWI', 'LST', 'wind_direction_mean'
            ]
            max_var = var_names[max_idx] if max_idx < len(var_names) else f"Unknown ({max_idx})"

            # FWI is channel 14. Let's see the center value (normalized)
            fwi_norm = input_tensor[0, 2, 14, 32, 32].item()
            rh_norm = input_tensor[0, 2, 9, 32, 32].item()

            print("\n[INFERENCE DEBUG] --- Datos de Entrada al Modelo ---")
            print(f"  Shape del Tensor    : {input_tensor.shape}")
            print(f"  Rango de Valores    : Min: {min_val:.4f} | Max: {max_val:.4f} (Var: {max_var})")
            print(f"  Media Global        : {mean_val:.4f}")
            print(f"  [CANARIO NORM] FWI Norm: {fwi_norm:.4f} | RH Norm: {rh_norm:.4f}")
            
            # --- CANARY LOGS (PHYSICAL) ---
            fwi_phys = features.get('features', {}).get('FWI', 0.0)
            rh_phys = features.get('features', {}).get('humidity', 0.0)
            print(f"  [CANARIO PHYS] FWI Real: {fwi_phys:.2f} | RH Real: {rh_phys:.2f}%")
            print("----------------------------------------------------\n")
            # ---------------------
            
            # The model is ConvLSTM (or similar). It takes (1, 3, 18, 64, 64).
            
            with torch.no_grad():
                out = model(input_tensor)
                
                # The robust model returns a dict with 'ignition' logits
                if isinstance(out, dict) and 'ignition' in out:
                    val = out['ignition'].item()
                elif isinstance(out, tuple) or isinstance(out, list):
                    val = out[0].item()
                else:
                    val = out.item()
                
                # Output logits need sigmoid to become probability [0, 1]
                prob = float(torch.sigmoid(torch.tensor(val)).item())
                
            if prob < 0.25:
                risk = "Low"
            elif prob < 0.50:
                risk = "Moderate"
            elif prob < 0.75:
                risk = "High"
            else:
                risk = "Extreme"
                
            print(f"[INFERENCE DEBUG] Logit en Bruto: {val:.4f} -> Probabilidad: {prob*100:.2f}% ({risk})")
                
        except Exception as e:
            print(f"Error executing model pass: {e}")
            return {"error": str(e)}
            
    return {
        "probability": float(prob),
        "risk_level": risk,
    }

def get_spread_model():
    global __spread_model
    if __spread_model is None:
        try:
            model_path = Path(__file__).resolve().parents[2] / "models" / "best_convlstm_v4_spread.pth"
            if model_path.exists():
                print(f"Loading ConvLSTM Spread model from {model_path}")
                from src.models.prop import RobustFireSpreadModel

                # NOTE: hardcoded 12 channels para modelos v4 legacy.
                # Cuando se retrenee con DEFAULT_FEATURE_VARS (29+1=30 ch),
                # cambiar a: input_channels=len(DEFAULT_FEATURE_VARS) + 1
                # (requiere extender src/api/data_fetcher.py para cargar las 18 vars nuevas).
                model = RobustFireSpreadModel(input_channels=12, hidden_dims=[64, 128])
                
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                __spread_model = model
            else:
                print(f"Warning: Spread Model not found at {model_path}.")
                __spread_model = "DUMMY"
        except Exception as e:
            print(f"Error loading spread model: {e}")
            __spread_model = "DUMMY"
    return __spread_model

async def predict_spread(tensor_np: np.ndarray, features: dict) -> dict:
    """
    Runs the spread model inference on the provided features.
    Converts the output mask into a GeoJSON Polygon.
    """
    if "error" in features:
        return {"error": features["error"]}

    model = get_spread_model()
    if model == "DUMMY" or model is None:
        return {"error": "Spread model not loaded. Weights may be missing."}
        
    try:
        device = next(model.parameters()).device
        input_tensor = torch.from_numpy(tensor_np).to(device)
        
        with torch.no_grad():
            channel_maxes = input_tensor.max(dim=0)[0].max(dim=0)[0].max(dim=1)[0].max(dim=1)[0]
            max_idx = torch.argmax(channel_maxes).item()
            
            var_names = [
                'elevation_mean', 'slope_mean', 'wind_u', 'wind_v',
                'hydric_stress', 'solar_risk', 'forest_prop', 'scrub_prop',
                'FWI', 'NDVI', 'dist_to_roads', 'fire_state'
            ]
            max_var = var_names[max_idx] if max_idx < len(var_names) else f"Unknown ({max_idx})"
            fwi_norm = input_tensor[0, 2, 8, 32, 32].item() # FWI is channel 8 in Spread
            fire_state_norm = input_tensor[0, 2, 11, 32, 32].item() # Fire state is channel 11

            print("\n[SPREAD DEBUG] --- Datos de Entrada al Modelo (ConvLSTM) ---")
            print(f"  Shape del Tensor    : {input_tensor.shape}")
            print(f"  Rango [Min, Max]    : [{input_tensor.min().item():.4f}, {input_tensor.max().item():.4f}]")
            print(f"  Variable Dominante  : {max_var} (Idx: {max_idx})")
            print(f"  [CANARIO NORM] FWI Norm: {fwi_norm:.4f} | Fire State Norm: {fire_state_norm:.4f}")
            
            # Sum of fire mask across all timesteps and pixels
            fire_sum = input_tensor[:, 2, 11, :, :].sum().item() if input_tensor.shape[1] == 3 else input_tensor[:, 11, :, :].sum().item()
            print(f"  [INTEGRITY] Total Fire Signal (Pixels): {fire_sum:.1f}")
            print("----------------------------------------------------------\n")

            out = model(input_tensor)
            
            # Handle different output shapes (B, T, 1, H, W) or (B, 1, H, W)
            if isinstance(out, dict):
                logits = out['spread_probability']
            else:
                logits = out
                
            # If we have multiple timesteps (T > 1), we take the MAX probability 
            # for each pixel across time. This gives the cumulative burned area.
            # IN THIS MODEL: T=3 represents 3 DAYS of propagation.
            if logits.dim() == 5: # (B, T, 1, H, W)
                spread_prob = torch.max(logits, dim=1)[0]
            else:
                spread_prob = logits
                
            # Squeeze to get a 2D map (H, W)
            spread_prob_2d = spread_prob.squeeze().cpu().numpy()
            
            print(f"[SPREAD DEBUG] Prediction Stats (3-Day Cumulative): min={spread_prob_2d.min():.4f}, max={spread_prob_2d.max():.4f}, mean={spread_prob_2d.mean():.4f}")
                
            # Binarize for the area calculation
            mask = (spread_prob_2d > 0.5).astype(np.uint8)
            
        # Convert map to GeoJSON drawing exact model pixels (1x1 km squares)
        geojson_polygon = []
        cell_bounds = features.get("cell_bounds", [])
        
        if len(cell_bounds) == 5:
            lats = [c[0] for c in cell_bounds[:-1]]
            lons = [c[1] for c in cell_bounds[:-1]]
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            
            lat_step = (max_lat - min_lat) / 64.0
            lon_step = (max_lon - min_lon) / 64.0
            
            for y_idx in range(64):
                for x_idx in range(64):
                    if mask[y_idx, x_idx] == 1:
                        # y_idx=0 is top (max_lat). y_idx increases downwards.
                        pixel_max_lat = max_lat - (y_idx * lat_step)
                        pixel_min_lat = pixel_max_lat - lat_step
                        pixel_min_lon = min_lon + (x_idx * lon_step)
                        pixel_max_lon = pixel_min_lon + lon_step
                        
                        poly = [
                            [float(pixel_min_lon), float(pixel_max_lat)], # top left
                            [float(pixel_max_lon), float(pixel_max_lat)], # top right
                            [float(pixel_max_lon), float(pixel_min_lat)], # bottom right
                            [float(pixel_min_lon), float(pixel_min_lat)], # bottom left
                            [float(pixel_min_lon), float(pixel_max_lat)]  # close loop
                        ]
                        geojson_polygon.append(poly)
                        
        # CALCULATE AREA: Only count each pixel once
        pixels_burned = int(mask.sum())
        # 1 pixel = 1km x 1km = 1,000,000 m2 = 100 hectares
        area_hectares = pixels_burned * 100
        
        # --- Cálculo del Rate of Spread (ROS) ---
        # Area en metros cuadrados: Ha * 10000
        area_m2 = area_hectares * 10000
        # Asumiendo propagación cuasi-circular: Área = PI * R^2 -> R = sqrt(Area / PI)
        radio_final_m = (area_m2 / 3.14159) ** 0.5

        # El área de ignición inicial era un cuadrado de 5x5 píxeles = 25 píxeles = 2500 Ha = 25000000 m2
        radio_inicial_m = (25 * 1000000 / 3.14159) ** 0.5

        # Distancia de avance del frente de llamas
        avance_frente_m = max(0, radio_final_m - radio_inicial_m)

        # El modelo predice propagación acumulada a 3 DÍAS (T=3 temporal context)
        duracion_minutos = 3 * 24 * 60  # 3 días = 4320 minutos
        ros_m_min = avance_frente_m / duracion_minutos if duracion_minutos > 0 else 0
        
        return {
            "geojson_polygon": geojson_polygon,
            "area_hectares": round(area_hectares, 2),
            "ros_m_min": round(ros_m_min, 2),
            "raw_mask_pixels": pixels_burned,
            "debug_max_prob": float(spread_prob.max().item())
        }
            
    except Exception as e:
        print(f"Error executing spread model: {e}")
        return {"error": str(e)}
