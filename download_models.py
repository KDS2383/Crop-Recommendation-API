import os
import gdown

print("📥 Downloading model files to models/ folder...")

os.makedirs("models", exist_ok=True)

files = {
    "tuned_augmented_crop_model.joblib": "1JwB_J4oUUcKqjof7TGbfBSVmSeB2zzo_",
    "original_feature_cols.joblib": "1Hpp_ORKZHd-yBCt5r2-VO_Kedzuz2xpk",
    "augmented_scaler.joblib": "1aeR0fNg8T4eQXO3rv2_znPMioDCmAcoe",
    "augmented_mlb.joblib": "1081Hz0tHcs2m4mRMi7YLuenTM0TkfYVe",
    "augmented_features_cache.parquet": "1QdGPdyWjbaQnH5x3CwSwYDnoTYmccRXr",
    "augmented_feature_cols.joblib": "1JwB_J4oUUcKqjof7TGbfBSVmSeB2zzo_"  # same as tuned model
}

for filename, file_id in files.items():
    output = f"/models/{filename}"
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"⬇️ Downloading {filename}...")
    gdown.download(url, output, quiet=False)

print("✅ All model files downloaded successfully!")
