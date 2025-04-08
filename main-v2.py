import fastapi
from fastapi import FastAPI, HTTPException # <<< ADD THIS LINE or make sure FastAPI is imported
import pandas as pd
import numpy as np
import joblib
import warnings
import time
import os
from typing import List, Optional

# Import Pydantic BaseModel
from pydantic import BaseModel

warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Pydantic Model for Input Validation ---
class UserInput(BaseModel):
    latitude: float
    longitude: float
    # --- Keys MUST match the incoming JSON ---
    # --- Assumption: Elevation WILL be included in the input JSON ---
    elevation: float
    soil_ph: float # Changed type to float, can be int too if always whole numbers
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
    # Use Optional if it might not be present, or provide a default
    historical_crops: List[str] = [] # Default to empty list if not provided

# --- Pydantic Model for Output ---
class RecommendationResponse(BaseModel):
    recommendations: List[str]
    processing_time_seconds: float

app = FastAPI()

# --- Configuration ---
MODEL_DIR = 'models'
DATA_DIR = 'data'

TUNED_MODEL_LOAD_NAME = os.path.join(MODEL_DIR, 'tuned_augmented_crop_model.joblib')
SCALER_LOAD_NAME = os.path.join(MODEL_DIR, 'augmented_scaler.joblib')
MLB_LOAD_NAME = os.path.join(MODEL_DIR, 'augmented_mlb.joblib')
ORIGINAL_FEATURE_COLS_LOAD_NAME = os.path.join(MODEL_DIR, 'original_feature_cols.joblib')
AUGMENTED_COLS_LOAD_NAME = os.path.join(MODEL_DIR, 'augmented_feature_cols.joblib')
CROP_CONDITIONS_FILE = os.path.join(DATA_DIR, 'crop_ideal_conditions_CLEANED_UNIQUE.csv')

# Tunable parameters
TOP_K_PREDICT = 20

PARAMETER_WEIGHTS = {
    # Critical factors (higher weight)
    'Avg_Temperature': 2.0,
    'Min_Temperature': 2.0,
    'Total_Rainfall': 2.5, # Often very critical
    'Soil_pH': 1.5,        # Important for nutrient availability
    'Soil_Moisture': 1.5,  # Critical water availability

    # Important factors (moderate weight)
    'Soil_Nitrogen': 1.0,
    'Soil_Phosphorus': 1.0,
    'Soil_Potassium': 1.0,
    'Avg_Humidity': 1.0,
    'Min_Humidity': 1.0,

    # Less critical factors (lower weight)
    'Soil_CEC': 0.75,
    'Avg_Wind_Speed': 0.5
}


# --- Global Variables: Load artifacts ONCE at startup ---
# (Copied from previous Flask version - uses global variables for simplicity)
print("Loading models and data...")
try:
    final_model = joblib.load(TUNED_MODEL_LOAD_NAME)
    scaler = joblib.load(SCALER_LOAD_NAME)
    mlb = joblib.load(MLB_LOAD_NAME)
    original_feature_cols = joblib.load(ORIGINAL_FEATURE_COLS_LOAD_NAME)
    augmented_feature_cols = joblib.load(AUGMENTED_COLS_LOAD_NAME)
    df_crop = pd.read_csv(CROP_CONDITIONS_FILE)
    df_crop.set_index('Crop', inplace=True)

    numeric_cols = [ # Ensure these match columns in df_crop
        'Soil_pH', 'Soil_Nitrogen', 'Soil_Phosphorus', 'Soil_Potassium',
        'Soil_Moisture', 'Soil_CEC', 'Avg_Temperature', 'Min_Temperature',
        'Avg_Humidity', 'Min_Humidity', 'Avg_Wind_Speed', 'Total_Rainfall',
        'Min_Elevation', 'Max_Elevation'
    ]
    for col in numeric_cols:
        if col in df_crop.columns:
            df_crop[col] = pd.to_numeric(df_crop[col], errors='coerce')

    param_cols_suitability = [ # Columns used for suitability calculation
        'Soil_pH', 'Soil_Nitrogen', 'Soil_Phosphorus', 'Soil_Potassium',
        'Soil_Moisture', 'Soil_CEC', 'Avg_Temperature', 'Min_Temperature',
        'Avg_Humidity', 'Min_Humidity', 'Avg_Wind_Speed', 'Total_Rainfall'
    ]
    missing_check = [col for col in param_cols_suitability if col not in df_crop.columns]
    if missing_check:
        raise ValueError(f"ERROR: Suitability columns missing in df_crop: {missing_check}.")
    else:
        param_ranges = df_crop[param_cols_suitability].max() - df_crop[param_cols_suitability].min() + 1e-6
        print("Calculated parameter ranges.")

    print("Models and data loaded successfully.")

except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not load required file: {e}. Ensure all paths are correct and files exist.")
    exit()
except Exception as e:
    print(f"FATAL ERROR during loading or setup: {e}")
    exit()


# === Helper Functions (Copied/Adapted - Debug prints removed/off) ===

# === Helper Functions ===

# --- Suitability Score Function (with WEIGHTS) ---
def calculate_suitability(loc_conditions, crop_ideal, param_cols, ranges, param_weights, # Added param_weights
                           elevation_penalty=100, crop_name_log="Unknown Crop", debug_print=False):
    """Calculates WEIGHTED dissimilarity score."""

    if debug_print: print(f"\n--- Calculating Suitability for {crop_name_log} ---")
    total_dissimilarity = 0.0 # Ensure float
    loc_elevation = loc_conditions.get('elevation', None)
    if debug_print: print(f"Loc Elevation: {loc_elevation} ({type(loc_elevation)})")

    # --- Elevation Check (remains the same logic, ensure float penalty) ---
    elevation_penalty_applied = False
    if loc_elevation is not None:
        min_elev = crop_ideal.get('Min_Elevation', None)
        max_elev = crop_ideal.get('Max_Elevation', None)
        if debug_print: print(f"Crop Elev Range: {min_elev} ({type(min_elev)}) - {max_elev} ({type(max_elev)})")
        if isinstance(loc_elevation, (int, float)):
             if not pd.isna(min_elev) and isinstance(min_elev, (int, float)) and loc_elevation < min_elev:
                 if debug_print: print("  * Elev Penalty (Low) APPLIED")
                 total_dissimilarity += float(elevation_penalty) # Ensure float
                 elevation_penalty_applied = True
             # Check Max Elev similarly
             if not pd.isna(max_elev) and isinstance(max_elev, (int, float)) and loc_elevation > max_elev:
                 if debug_print: print("  * Elev Penalty (High) APPLIED")
                 total_dissimilarity += float(elevation_penalty) # Ensure float
                 elevation_penalty_applied = True
             # Add else for debug print if checks passed
             elif debug_print and not elevation_penalty_applied : print("  * Elev Checks PASSED")
        else:
             if debug_print: print(f"  * Skipping Elev Check (loc_elevation type: {type(loc_elevation)})")


    # --- Parameter Comparison ---
    skipped_count = 0
    calculated_count = 0
    for param in param_cols:
        loc_val = loc_conditions.get(param, None)
        ideal_val = crop_ideal.get(param, None)
        e_print = None # Error flag

        # --- Safe Print (Conditional) ---
        if debug_print:
            try:
                loc_str = loc_val if loc_val is not None else "N/A"
                ideal_str = ideal_val if ideal_val is not None else "N/A"
                if isinstance(ideal_str, dict): ideal_str = "{Dict}"
                if isinstance(loc_str, dict): loc_str = "{Dict}"
                print(f"  Param: {param:<15} | Loc: {loc_str!s:<8} | Ideal: {ideal_str!s:<8}", end='') # Use !s for safety
            except TypeError as e_print:
                 print(f"\nDEBUG: Print Formatting TypeError for Param '{param}'!") # ... (rest of error logging)
                 print(" | Skipping...", end='')


        # Check if both values are valid numbers
        if loc_val is not None and ideal_val is not None and \
           not pd.isna(loc_val) and not pd.isna(ideal_val) and \
           isinstance(loc_val, (int, float)) and isinstance(ideal_val, (int, float)):

            calculated_count += 1
            param_range = ranges.get(param, 1e-6)
            difference = abs(loc_val - ideal_val)
            normalized_diff = difference / param_range

            # --- Apply Weight ---
            weight = param_weights.get(param, 1.0) # Default to 1.0 if param not in weights dict
            weighted_diff = normalized_diff * weight
            total_dissimilarity += weighted_diff # Add weighted difference
            # ---

            if debug_print:
                # (Update safe print for numeric results to show weight and weighted diff)
                try:
                     diff_str = f"{difference:<8.2f}" if isinstance(difference, (int, float)) else "N/A"
                     range_str = f"{param_range:<8.2f}" if isinstance(param_range, (int, float)) else "N/A"
                     norm_str = f"{normalized_diff:.4f}" if isinstance(normalized_diff, (int, float)) else "N/A"
                     weight_str = f"{weight:.2f}"
                     wdiff_str = f"{weighted_diff:.4f}"
                     print(f" | Diff: {diff_str} | Range: {range_str} | NormDiff: {norm_str} | Wgt: {weight_str} | WgtDiff: {wdiff_str}")
                except TypeError as e_print2:
                     # ... (error logging)
                     pass
        else:
            skipped_count += 1
            if debug_print and e_print is None: # Only print if first print didn't fail
                 print(" | Skipping (Missing/Invalid)")

    if debug_print: # Update summary print
        print(f"--- Summary for {crop_name_log} ---")
        print(f"  Params Calculated: {calculated_count}")
        print(f"  Params Skipped:    {skipped_count}")
        print(f"  Elevation Penalty Applied: {elevation_penalty_applied}")
        print(f"--- Final Weighted Total Dissimilarity: {total_dissimilarity:.4f} ---")

    return total_dissimilarity

def predict_probabilities_augmented(user_input_dict, model, scaler, df_crop_global,
                                     original_feature_cols_global, param_cols_suitability_global,
                                     param_ranges_global, augmented_cols_global, mlb,
                                     param_weights): # <-- Added param_weights
    """Calculates augmented features and returns probability dictionary."""
    user_suitability_scores = {}
    for crop_name, ideal_conditions_series in df_crop_global.iterrows():
        ideal_conditions_dict = ideal_conditions_series.to_dict()
        score = calculate_suitability( # Pass weights and debug_print=False
            user_input_dict, ideal_conditions_dict, param_cols_suitability_global,
            param_ranges_global, param_weights, # <-- Pass weights
            crop_name_log=crop_name, debug_print=False
        )
        user_suitability_scores[f"suitability_{crop_name}"] = score
    # --- Rest is unchanged ---
    user_feature_vector_list = []
    for col in original_feature_cols_global:
        user_feature_vector_list.append(user_input_dict.get(col, np.nan))
    suitability_col_names = [col for col in augmented_cols_global if col.startswith('suitability_')]
    for col_name in suitability_col_names:
         user_feature_vector_list.append(user_suitability_scores.get(col_name, np.nan))
    user_df_temp = pd.DataFrame([dict(zip(augmented_cols_global, user_feature_vector_list))], columns=augmented_cols_global)
    user_df_temp.fillna(0, inplace=True)
    user_vector_scaled = scaler.transform(user_df_temp)
    proba_list = model.predict_proba(user_vector_scaled)
    crop_probabilities = np.array([p[:, 1] for p in proba_list]).T[0]
    probability_dict = dict(zip(mlb.classes_, crop_probabilities))
    return probability_dict

def recommend_predict_then_validate(user_input_dict, top_k, model, scaler, mlb,
                                    df_crop_global, original_feature_cols_global,
                                    param_cols_suitability_global, param_ranges_global,
                                    augmented_cols_global, param_weights, # <-- Added param_weights
                                    exclude_crops: List[str] = []):
    """
    Recommends crops: top K ML preds, re-rank by suitability, excluding specified crops.
    """
    try:
        # Pass weights down
        ml_probabilities = predict_probabilities_augmented(
            user_input_dict, model, scaler, df_crop_global,
            original_feature_cols_global, param_cols_suitability_global,
            param_ranges_global, augmented_cols_global, mlb,
            param_weights # <-- Pass weights
        )
    except Exception as e:
        print(f"Error during probability prediction: {e}")
        return []

    sorted_ml_preds = sorted(ml_probabilities.items(), key=lambda item: item[1], reverse=True)
    top_k_ml_crops = dict(sorted_ml_preds[:top_k])

    if not top_k_ml_crops: return []

    validated_candidates = []
    for crop_name, ml_prob in top_k_ml_crops.items():
        if crop_name in df_crop_global.index:
            ideal_conditions_series = df_crop_global.loc[crop_name]
            ideal_conditions_dict = ideal_conditions_series.to_dict()
            try:
                suitability_score = calculate_suitability( # Pass weights
                    user_input_dict, ideal_conditions_dict, param_cols_suitability_global,
                    param_ranges_global, param_weights, # <-- Pass weights
                    crop_name_log=crop_name, debug_print=False
                )
                validated_candidates.append({
                    'crop': crop_name, 'ml_prob': ml_prob,
                    'suitability_score': suitability_score
                })
            except Exception as e_calc:
                 print(f"Error calculating suitability for {crop_name}: {e_calc}")

    # Sort Primarily by Suitability (Ascending), then ML Prob (Descending)
    validated_candidates.sort(key=lambda x: (x['suitability_score'], -x['ml_prob']))

    # Filter out excluded crops (unchanged)
    filtered_candidates = []
    if exclude_crops:
        exclude_crops_set = set(exclude_crops)
        print(f"Excluding historical crops: {exclude_crops_set}")
        for candidate in validated_candidates:
            if candidate['crop'] not in exclude_crops_set:
                filtered_candidates.append(candidate)
        print(f"Candidates remaining after filtering: {len(filtered_candidates)}")
    else:
        filtered_candidates = validated_candidates

    # Select top 3 from the FILTERED list (unchanged)
    top_3 = [item['crop'] for item in filtered_candidates[:3]]

    # Optional Logging (unchanged)
    print("\nTop candidates BEFORE filtering historical crops:")
    for item in validated_candidates[:10]:
         print(f"  - {item['crop']}: Suitability={item['suitability_score']:.4f}, Prob={item['ml_prob']:.4f}")
    print("\nTop candidates AFTER filtering historical crops:")
    for item in filtered_candidates[:10]:
         print(f"  - {item['crop']}: Suitability={item['suitability_score']:.4f}, Prob={item['ml_prob']:.4f}")

    return top_3


# === FastAPI Endpoint ===

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(user_input: UserInput):
    """
    Receives location, soil, weather data and returns top 3 crop recommendations,
    excluding historical crops.
    """
    start_time = time.time()
    try:
        # Map input keys (unchanged)
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

        # Prepare shared arguments (using globally loaded objects)
        shared_args = {
            'model': final_model, 'scaler': scaler, 'mlb': mlb,
            'df_crop_global': df_crop, 'original_feature_cols_global': original_feature_cols,
            'param_cols_suitability_global': param_cols_suitability,
            'param_ranges_global': param_ranges,
            'augmented_cols_global': augmented_feature_cols,
            'param_weights': PARAMETER_WEIGHTS # <-- Add this line
        }

        # --- Call the recommendation logic, PASSING historical_crops ---
        recommendations = recommend_predict_then_validate(
            user_input_dict=user_conditions_dict,
            top_k=TOP_K_PREDICT,
            exclude_crops=user_input.historical_crops, # Pass the list here
            **shared_args
        )

        end_time = time.time()
        processing_time = end_time - start_time

        if not recommendations:
             raise fastapi.HTTPException(status_code=500, detail="Could not generate recommendations")

        # --- Check if we have fewer than 3 recommendations after filtering ---
        if len(recommendations) < 3:
            print(f"Warning: Only {len(recommendations)} recommendations generated after excluding historical crops.")
            # Decide how to handle this: return fewer, or perhaps pad with next best?
            # Current behavior returns the list as is (potentially fewer than 3).

        return RecommendationResponse(
            recommendations=recommendations,
            processing_time_seconds=round(processing_time, 3)
        )

    except Exception as e:
        print(f"Error processing /recommend request: {e}")
        import traceback
        traceback.print_exc()
        raise fastapi.HTTPException(status_code=500, detail="Internal server error processing request")


@app.get("/")
async def root():
    return {"message": "Crop Recommendation API is running. POST to /recommend."}


# uvicorn main:app --reload --host 0.0.0.0 --port 8000
# Or: python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000    