# --- START OF FILE main.py (Option 2 Implemented) ---

import fastapi
from fastapi import FastAPI, HTTPException # Ensure FastAPI and HTTPException are imported
import pandas as pd
import numpy as np
import joblib
import warnings
import time
import os
from typing import List, Optional
import threading # Import threading for the lock
import psutil

# Import Pydantic BaseModel
from pydantic import BaseModel

warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Pydantic Models (Unchanged) ---
class UserInput(BaseModel):
    latitude: float
    longitude: float
    elevation: float
    soil_ph: float
    soil_nitrogen: int
    soil_phosphorus: int
    soil_potassium: int
    soil_moisture: int
    soil_cec: int
    avg_temperature: float
    min_temperature: float
    avg_humidity: float
    min_humidity: float
    avg_wind_speed: float
    total_rainfall: float
    historical_crops: List[str] = []

class RecommendationResponse(BaseModel):
    recommendations: List[str]
    processing_time_seconds: float

app = FastAPI()

# --- Configuration (Unchanged) ---
MODEL_DIR = 'models'
DATA_DIR = 'data'

TUNED_MODEL_LOAD_NAME = os.path.join(MODEL_DIR, 'tuned_augmented_crop_model.joblib')
SCALER_LOAD_NAME = os.path.join(MODEL_DIR, 'augmented_scaler.joblib')
MLB_LOAD_NAME = os.path.join(MODEL_DIR, 'augmented_mlb.joblib')
ORIGINAL_FEATURE_COLS_LOAD_NAME = os.path.join(MODEL_DIR, 'original_feature_cols.joblib')
AUGMENTED_COLS_LOAD_NAME = os.path.join(MODEL_DIR, 'augmented_feature_cols.joblib')
CROP_CONDITIONS_FILE = os.path.join(DATA_DIR, 'crop_ideal_conditions_CLEANED_UNIQUE.csv')

TOP_K_PREDICT = 20
PARAMETER_WEIGHTS = {
    'Avg_Temperature': 2.0, 'Min_Temperature': 2.0, 'Total_Rainfall': 2.5,
    'Soil_pH': 1.5, 'Soil_Moisture': 1.5, 'Soil_Nitrogen': 1.0,
    'Soil_Phosphorus': 1.0, 'Soil_Potassium': 1.0, 'Avg_Humidity': 1.0,
    'Min_Humidity': 1.0, 'Soil_CEC': 0.75, 'Avg_Wind_Speed': 0.5
}


# --- Get memory BEFORE loading ---
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / (1024 * 1024) # Resident Set Size in MiB
print(f"Memory before loading: {mem_before:.2f} MiB")


# --- Global Variables: Load non-model artifacts at startup ---
print("Loading non-model artifacts and data...")
try:
    # Load everything EXCEPT the main model
    scaler = joblib.load(SCALER_LOAD_NAME)
    mlb = joblib.load(MLB_LOAD_NAME)
    original_feature_cols = joblib.load(ORIGINAL_FEATURE_COLS_LOAD_NAME)
    augmented_feature_cols = joblib.load(AUGMENTED_COLS_LOAD_NAME)

    # --- Load df_crop (applying optional optimizations) ---
    print("Optimizing df_crop loading...")
    param_cols_suitability = [
        'Soil_pH', 'Soil_Nitrogen', 'Soil_Phosphorus', 'Soil_Potassium',
        'Soil_Moisture', 'Soil_CEC', 'Avg_Temperature', 'Min_Temperature',
        'Avg_Humidity', 'Min_Humidity', 'Avg_Wind_Speed', 'Total_Rainfall'
    ]
    cols_to_load = ['Crop', 'Min_Elevation', 'Max_Elevation'] + param_cols_suitability
    dtype_map = { # Example dtypes - adjust if needed
        col: 'float32' for col in cols_to_load if col not in ['Crop', 'Soil_Nitrogen', 'Soil_Phosphorus', 'Soil_Potassium', 'Soil_Moisture', 'Soil_CEC']
    }
    dtype_map.update({ # Integers if appropriate
        'Soil_Nitrogen': 'int16', 'Soil_Phosphorus': 'int16', 'Soil_Potassium': 'int16',
        'Soil_Moisture': 'int16', 'Soil_CEC': 'int16',
        # Keep elevations float unless always whole numbers
        'Min_Elevation': 'float32', 'Max_Elevation': 'float32'
    })

    df_crop = pd.read_csv(
        CROP_CONDITIONS_FILE,
        usecols=lambda c: c in cols_to_load, # Use lambda to handle potential missing cols robustly
        index_col='Crop',
        dtype={k: v for k, v in dtype_map.items() if k != 'Crop'} # Apply relevant dtypes
    )
    print(f"df_crop loaded. Memory usage:")
    print(df_crop.info(memory_usage='deep'))

    # --- Ensure numeric types and calculate ranges (Unchanged logic) ---
    numeric_cols = [
        'Soil_pH', 'Soil_Nitrogen', 'Soil_Phosphorus', 'Soil_Potassium',
        'Soil_Moisture', 'Soil_CEC', 'Avg_Temperature', 'Min_Temperature',
        'Avg_Humidity', 'Min_Humidity', 'Avg_Wind_Speed', 'Total_Rainfall',
        'Min_Elevation', 'Max_Elevation'
    ]
    for col in numeric_cols:
        if col in df_crop.columns:
            df_crop[col] = pd.to_numeric(df_crop[col], errors='coerce')

    missing_check = [col for col in param_cols_suitability if col not in df_crop.columns]
    if missing_check:
        raise ValueError(f"ERROR: Suitability columns missing in df_crop: {missing_check}.")
    else:
        param_ranges = df_crop[param_cols_suitability].max() - df_crop[param_cols_suitability].min() + 1e-6
        print("Calculated parameter ranges.")

    print("Non-model artifacts and data loaded successfully.")
    
    # --- Get memory AFTER loading ---
    mem_after = process.memory_info().rss / (1024 * 1024) # RSS in MiB
    print(f"Memory after loading non-model artifacts: {mem_after:.2f} MiB")
    print(f"Startup memory usage (approx): {mem_after - mem_before:.2f} MiB")


except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not load required file during startup: {e}.")
    # Optionally raise the error to prevent the app from starting partially
    # raise RuntimeError(f"FATAL ERROR: Could not load required file during startup: {e}") from e
    scaler = mlb = original_feature_cols = augmented_feature_cols = df_crop = param_ranges = None # Indicate failure
    # Or simply exit() if preferred for critical failure
    # exit(1)

except Exception as e:
    print(f"FATAL ERROR during non-model loading or setup: {e}")
    import traceback
    traceback.print_exc()
    scaler = mlb = original_feature_cols = augmented_feature_cols = df_crop = param_ranges = None # Indicate failure
    # exit(1)


# --- Model variable, initially None ---
final_model = None
# Use a lock to prevent multiple requests trying to load the model simultaneously
model_load_lock = threading.Lock()

def get_model():
    """Loads the XGBoost model if it hasn't been loaded yet. Thread-safe."""
    global final_model
    if final_model is None: # Quick check outside the lock for efficiency
        with model_load_lock: # Acquire lock before potentially loading
            if final_model is None: # Double-check inside the lock
                print("Loading XGBoost model ON DEMAND...")
                load_start_time = time.time()
                try:
                    final_model = joblib.load(TUNED_MODEL_LOAD_NAME)
                    load_end_time = time.time()
                    print(f"XGBoost model loaded successfully in {load_end_time - load_start_time:.2f} seconds.")
                except FileNotFoundError:
                    print(f"ERROR: Model file not found at {TUNED_MODEL_LOAD_NAME}")
                    raise HTTPException(status_code=503, detail="Model file not found, cannot make predictions.")
                except Exception as e:
                    print(f"ERROR: Failed to load model: {e}")
                    raise HTTPException(status_code=503, detail=f"Failed to load model: {e}")

    # Check again in case loading failed inside the lock but didn't raise for some reason
    if final_model is None:
        print("ERROR: Model is still None after attempting load.")
        raise HTTPException(status_code=503, detail="Model unavailable after load attempt.")

    return final_model

# === Helper Functions (Adjusted for On-Demand Model Loading) ===

# --- Suitability Score Function (Unchanged) ---
def calculate_suitability(loc_conditions, crop_ideal, param_cols, ranges, param_weights,
                           elevation_penalty=100, crop_name_log="Unknown Crop", debug_print=False):
    # ... (This function remains exactly the same as before) ...
    if debug_print: print(f"\n--- Calculating Suitability for {crop_name_log} ---")
    total_dissimilarity = 0.0
    loc_elevation = loc_conditions.get('elevation', None)
    if debug_print: print(f"Loc Elevation: {loc_elevation} ({type(loc_elevation)})")
    elevation_penalty_applied = False
    if loc_elevation is not None:
        min_elev = crop_ideal.get('Min_Elevation', None)
        max_elev = crop_ideal.get('Max_Elevation', None)
        if debug_print: print(f"Crop Elev Range: {min_elev} ({type(min_elev)}) - {max_elev} ({type(max_elev)})")
        if isinstance(loc_elevation, (int, float)):
            if not pd.isna(min_elev) and isinstance(min_elev, (int, float)) and loc_elevation < min_elev:
                if debug_print: print("  * Elev Penalty (Low) APPLIED")
                total_dissimilarity += float(elevation_penalty)
                elevation_penalty_applied = True
            if not pd.isna(max_elev) and isinstance(max_elev, (int, float)) and loc_elevation > max_elev:
                if debug_print: print("  * Elev Penalty (High) APPLIED")
                total_dissimilarity += float(elevation_penalty)
                elevation_penalty_applied = True
            elif debug_print and not elevation_penalty_applied : print("  * Elev Checks PASSED")
        else:
            if debug_print: print(f"  * Skipping Elev Check (loc_elevation type: {type(loc_elevation)})")
    skipped_count = 0
    calculated_count = 0
    for param in param_cols:
        loc_val = loc_conditions.get(param, None)
        ideal_val = crop_ideal.get(param, None)
        e_print = None
        if debug_print:
            try:
                loc_str = loc_val if loc_val is not None else "N/A"; ideal_str = ideal_val if ideal_val is not None else "N/A"
                if isinstance(ideal_str, dict): ideal_str = "{Dict}"; 
                if isinstance(loc_str, dict): loc_str = "{Dict}"
                print(f"  Param: {param:<15} | Loc: {loc_str!s:<8} | Ideal: {ideal_str!s:<8}", end='')
            except TypeError as e_print: print(f"\nDEBUG: Print Formatting TypeError for Param '{param}'!") ; print(" | Skipping...", end='')
        if loc_val is not None and ideal_val is not None and not pd.isna(loc_val) and not pd.isna(ideal_val) and isinstance(loc_val, (int, float)) and isinstance(ideal_val, (int, float)):
            calculated_count += 1; param_range = ranges.get(param, 1e-6); difference = abs(loc_val - ideal_val); normalized_diff = difference / param_range
            weight = param_weights.get(param, 1.0); weighted_diff = normalized_diff * weight; total_dissimilarity += weighted_diff
            if debug_print:
                try:
                     diff_str = f"{difference:<8.2f}" if isinstance(difference, (int, float)) else "N/A"; range_str = f"{param_range:<8.2f}" if isinstance(param_range, (int, float)) else "N/A"; norm_str = f"{normalized_diff:.4f}" if isinstance(normalized_diff, (int, float)) else "N/A"; weight_str = f"{weight:.2f}"; wdiff_str = f"{weighted_diff:.4f}"
                     print(f" | Diff: {diff_str} | Range: {range_str} | NormDiff: {norm_str} | Wgt: {weight_str} | WgtDiff: {wdiff_str}")
                except TypeError as e_print2: pass
        else: skipped_count += 1
        if debug_print and e_print is None: print(" | Skipping (Missing/Invalid)")
    if debug_print: print(f"--- Summary for {crop_name_log} ---"); print(f"  Params Calculated: {calculated_count}"); print(f"  Params Skipped:    {skipped_count}"); print(f"  Elevation Penalty Applied: {elevation_penalty_applied}"); print(f"--- Final Weighted Total Dissimilarity: {total_dissimilarity:.4f} ---")
    return total_dissimilarity

def predict_probabilities_augmented(user_input_dict, scaler, df_crop_global, # <<< REMOVED model arg
                                     original_feature_cols_global, param_cols_suitability_global,
                                     param_ranges_global, augmented_cols_global, mlb,
                                     param_weights):
    """Calculates augmented features and returns probability dictionary."""
    model = get_model() # <<< GET model here

    user_suitability_scores = {}
    for crop_name, ideal_conditions_series in df_crop_global.iterrows():
        ideal_conditions_dict = ideal_conditions_series.to_dict()
        score = calculate_suitability(
            user_input_dict, ideal_conditions_dict, param_cols_suitability_global,
            param_ranges_global, param_weights,
            crop_name_log=crop_name, debug_print=False
        )
        user_suitability_scores[f"suitability_{crop_name}"] = score

    # --- Prepare feature vector (Unchanged logic) ---
    user_feature_vector_list = []
    for col in original_feature_cols_global:
        user_feature_vector_list.append(user_input_dict.get(col, np.nan))
    suitability_col_names = [col for col in augmented_cols_global if col.startswith('suitability_')]
    for col_name in suitability_col_names:
         user_feature_vector_list.append(user_suitability_scores.get(col_name, np.nan))

    user_df_temp = pd.DataFrame([dict(zip(augmented_cols_global, user_feature_vector_list))], columns=augmented_cols_global)
    user_df_temp.fillna(0, inplace=True) # Consider more sophisticated imputation if needed

    # Scale and Predict
    user_vector_scaled = scaler.transform(user_df_temp)
    proba_list = model.predict_proba(user_vector_scaled) # Use the loaded model

    # Extract probabilities
    crop_probabilities = np.array([p[:, 1] for p in proba_list]).T[0]
    probability_dict = dict(zip(mlb.classes_, crop_probabilities))
    return probability_dict

def recommend_predict_then_validate(user_input_dict, top_k, scaler, mlb, # <<< REMOVED model arg
                                    df_crop_global, original_feature_cols_global,
                                    param_cols_suitability_global, param_ranges_global,
                                    augmented_cols_global, param_weights,
                                    exclude_crops: List[str] = []):
    """
    Recommends crops: top K ML preds, re-rank by suitability, excluding specified crops.
    """
    # Optional: Call get_model() here if you want to ensure it's loaded early
    # _ = get_model()
    try:
        # Call predict_probabilities_augmented WITHOUT model argument
        ml_probabilities = predict_probabilities_augmented(
            user_input_dict, scaler, df_crop_global,
            original_feature_cols_global, param_cols_suitability_global,
            param_ranges_global, augmented_cols_global, mlb,
            param_weights
        )
    except HTTPException as http_exc: # Catch potential model loading errors
         print(f"HTTP Exception during probability prediction (likely model load issue): {http_exc.detail}")
         raise # Re-raise the exception to be caught by the endpoint handler
    except Exception as e:
        print(f"Error during probability prediction: {e}")
        # Avoid returning empty list directly, raise an error for endpoint to handle
        raise RuntimeError(f"Probability prediction failed: {e}") from e


    sorted_ml_preds = sorted(ml_probabilities.items(), key=lambda item: item[1], reverse=True)
    top_k_ml_crops = dict(sorted_ml_preds[:top_k])

    if not top_k_ml_crops:
        print("Warning: No crops found in top K ML predictions.")
        return [] # It's okay to return empty if ML gives nothing

    # --- Validation and Re-ranking (Unchanged logic) ---
    validated_candidates = []
    for crop_name, ml_prob in top_k_ml_crops.items():
        if crop_name in df_crop_global.index:
            ideal_conditions_series = df_crop_global.loc[crop_name]
            ideal_conditions_dict = ideal_conditions_series.to_dict()
            try:
                suitability_score = calculate_suitability(
                    user_input_dict, ideal_conditions_dict, param_cols_suitability_global,
                    param_ranges_global, param_weights,
                    crop_name_log=crop_name, debug_print=False
                )
                validated_candidates.append({
                    'crop': crop_name, 'ml_prob': ml_prob,
                    'suitability_score': suitability_score
                })
            except Exception as e_calc:
                 print(f"Error calculating suitability for {crop_name}: {e_calc}")
                 # Decide how to handle: skip candidate or assign high penalty score?
                 # Skipping for now:
                 continue

    # Sort Primarily by Suitability (Ascending), then ML Prob (Descending)
    validated_candidates.sort(key=lambda x: (x['suitability_score'], -x['ml_prob']))

    # --- Filtering (Unchanged logic) ---
    filtered_candidates = []
    if exclude_crops:
        exclude_crops_set = set(exclude_crops)
        print(f"Excluding historical crops: {exclude_crops_set}")
        filtered_candidates = [c for c in validated_candidates if c['crop'] not in exclude_crops_set]
        print(f"Candidates remaining after filtering: {len(filtered_candidates)}")
    else:
        filtered_candidates = validated_candidates

    top_3 = [item['crop'] for item in filtered_candidates[:3]]

    # --- Logging (Unchanged) ---
    print("\nTop candidates BEFORE filtering historical crops:")
    for item in validated_candidates[:10]: print(f"  - {item['crop']}: Suitability={item['suitability_score']:.4f}, Prob={item['ml_prob']:.4f}")
    print("\nTop candidates AFTER filtering historical crops:")
    for item in filtered_candidates[:10]: print(f"  - {item['crop']}: Suitability={item['suitability_score']:.4f}, Prob={item['ml_prob']:.4f}")

    return top_3

@app.get("/proxy-image")
async def proxy_image(url: str):
    """
    Proxies external crop images to avoid CORS issues for frontend rendering & PDF generation.
    Usage: /proxy-image?url=https://example.com/image.jpg
    """
    try:
        # Use a timeout to prevent requests from hanging indefinitely
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            # Raise an error for non-200 responses
            response.raise_for_status() 
            
            content_type = response.headers.get("Content-Type", "image/jpeg")
            return StreamingResponse(response.aiter_bytes(), media_type=content_type)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch image from source: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error proxying image: {str(e)}")
        
# === FastAPI Endpoint (Adjusted) ===

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(user_input: UserInput):
    """
    Receives location, soil, weather data and returns top 3 crop recommendations,
    excluding historical crops. Model is loaded on first request.
    """
    start_time = time.time()
    try:
        # --- Check if essential non-model artifacts loaded okay ---
        # These checks are important if startup loading could have failed
        if scaler is None or mlb is None or df_crop is None or param_ranges is None:
             raise HTTPException(status_code=503, detail="Server configuration error: Core data/artifacts not loaded.")

        # Optional: Trigger model load early if desired, otherwise helpers will do it.
        # _ = get_model()

        # --- Map input keys (Unchanged) ---
        user_conditions_dict = {
            "latitude": user_input.latitude, "longitude": user_input.longitude,
            "elevation": user_input.elevation, "Soil_pH": user_input.soil_ph,
            "Soil_Nitrogen": user_input.soil_nitrogen, "Soil_Phosphorus": user_input.soil_phosphorus,
            "Soil_Potassium": user_input.soil_potassium, "Soil_Moisture": user_input.soil_moisture,
            "Soil_CEC": user_input.soil_cec, "Avg_Temperature": user_input.avg_temperature,
            "Min_Temperature": user_input.min_temperature, "Avg_Humidity": user_input.avg_humidity,
            "Min_Humidity": user_input.min_humidity, "Avg_Wind_Speed": user_input.avg_wind_speed,
            "Total_Rainfall": user_input.total_rainfall
        }

        # Prepare shared arguments (using globally loaded objects - NO MODEL HERE)
        shared_args = {
            'scaler': scaler, 'mlb': mlb, # <<< NO 'model' key
            'df_crop_global': df_crop, 'original_feature_cols_global': original_feature_cols,
            'param_cols_suitability_global': param_cols_suitability,
            'param_ranges_global': param_ranges,
            'augmented_cols_global': augmented_feature_cols,
            'param_weights': PARAMETER_WEIGHTS
        }

        # --- Call the recommendation logic (WITHOUT model argument) ---
        recommendations = recommend_predict_then_validate(
            user_input_dict=user_conditions_dict,
            top_k=TOP_K_PREDICT,
            exclude_crops=user_input.historical_crops,
            **shared_args # Pass args without model
        )

        end_time = time.time()
        processing_time = end_time - start_time

        # --- Handle no recommendations ---
        # recommend_predict_then_validate now returns empty list in normal "no result" cases
        # It raises exceptions for actual errors (like model load failure).
        if not recommendations:
             print("No suitable recommendations found after filtering.")
             # Return empty list in the response, maybe with a specific message later if needed
             # recommendations = [] # Already is empty list

        # Check length after filtering (Unchanged)
        if recommendations and len(recommendations) < 3:
            print(f"Warning: Only {len(recommendations)} recommendations generated after excluding historical crops.")

        return RecommendationResponse(
            recommendations=recommendations, # Return empty list if none found
            processing_time_seconds=round(processing_time, 3)
        )

    except HTTPException as http_exc:
        # Catch exceptions specifically raised for client errors or model load issues
        print(f"HTTP Exception in /recommend: {http_exc.detail}")
        raise http_exc # Re-raise to let FastAPI handle it
    except Exception as e:
        # Catch unexpected errors during processing
        print(f"Unexpected Error processing /recommend request: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/")
async def root():
    # Optionally check if model has been loaded here for a status endpoint
    # model_status = "Loaded" if final_model is not None else "Not Loaded Yet"
    # return {"message": "Crop Recommendation API running.", "model_status": model_status}
    return {"message": "Crop Recommendation API is running. POST to /recommend."}

# --- uvicorn command (Unchanged) ---
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
# python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
# Or for Render/Deta: uvicorn main:app --host 0.0.0.0 --port $PORT

# --- END OF FILE main.py (Option 2 Implemented) ---
