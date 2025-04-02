import requests
import json
import time

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/recommend" # Your running FastAPI endpoint
REQUEST_TIMEOUT = 60 # Seconds to wait for a response

# --- List of 10 Location Payloads ---
test_payloads = [
    # 1: Amritsar, Punjab
    {"location_name": "Amritsar, Punjab", "data": {
        "latitude": 31.6340, "longitude": 74.8723, "elevation": 230.0, "soil_ph": 7.5,
        "soil_nitrogen": 260, "soil_phosphorus": 180, "soil_potassium": 240, "soil_moisture": 380,
        "soil_cec": 250, "avg_temperature": 25.0, "min_temperature": 10.0, "avg_humidity": 60.0,
        "min_humidity": 25.0, "avg_wind_speed": 8.0, "total_rainfall": 700.0, "historical_crops": ["Wheat", "Rice"]
    }},
    # 2: Ratnagiri, MH
    {"location_name": "Ratnagiri, MH", "data": {
        "latitude": 16.9944, "longitude": 73.3000, "elevation": 11.0, "soil_ph": 6.2,
        "soil_nitrogen": 220, "soil_phosphorus": 150, "soil_potassium": 200, "soil_moisture": 450,
        "soil_cec": 200, "avg_temperature": 28.0, "min_temperature": 21.0, "avg_humidity": 80.0,
        "min_humidity": 60.0, "avg_wind_speed": 12.0, "total_rainfall": 3000.0, "historical_crops": ["Mango", "Rice", "Cashew"]
    }},
    # 3: Jaisalmer, Raj.
    {"location_name": "Jaisalmer, Raj.", "data": {
        "latitude": 26.9157, "longitude": 70.9083, "elevation": 225.0, "soil_ph": 8.0,
        "soil_nitrogen": 150, "soil_phosphorus": 100, "soil_potassium": 180, "soil_moisture": 200,
        "soil_cec": 150, "avg_temperature": 30.0, "min_temperature": 15.0, "avg_humidity": 40.0,
        "min_humidity": 10.0, "avg_wind_speed": 15.0, "total_rainfall": 250.0, "historical_crops": ["Pearl Millet", "Moth Bean"]
    }},
    # 4: Hyderabad, Deccan
    {"location_name": "Hyderabad, Deccan", "data": {
        "latitude": 17.3850, "longitude": 78.4867, "elevation": 542.0, "soil_ph": 7.0,
        "soil_nitrogen": 200, "soil_phosphorus": 140, "soil_potassium": 210, "soil_moisture": 300,
        "soil_cec": 220, "avg_temperature": 28.5, "min_temperature": 18.0, "avg_humidity": 65.0,
        "min_humidity": 30.0, "avg_wind_speed": 9.0, "total_rainfall": 800.0, "historical_crops": ["Sorghum", "Cotton", "Maize"]
    }},
    # 5: Shillong, Megh.
    {"location_name": "Shillong, Megh.", "data": {
        "latitude": 25.5788, "longitude": 91.8933, "elevation": 1525.0, "soil_ph": 5.5,
        "soil_nitrogen": 280, "soil_phosphorus": 160, "soil_potassium": 190, "soil_moisture": 500,
        "soil_cec": 230, "avg_temperature": 18.0, "min_temperature": 8.0, "avg_humidity": 85.0,
        "min_humidity": 65.0, "avg_wind_speed": 5.0, "total_rainfall": 2500.0, "historical_crops": ["Potato", "Ginger", "Turmeric"]
    }},
    # 6: Patna, Bihar
    {"location_name": "Patna, Bihar", "data": {
        "latitude": 25.5941, "longitude": 85.1376, "elevation": 53.0, "soil_ph": 7.2,
        "soil_nitrogen": 240, "soil_phosphorus": 170, "soil_potassium": 230, "soil_moisture": 400,
        "soil_cec": 240, "avg_temperature": 27.0, "min_temperature": 12.0, "avg_humidity": 70.0,
        "min_humidity": 40.0, "avg_wind_speed": 7.0, "total_rainfall": 1100.0, "historical_crops": ["Rice", "Wheat", "Lentil"]
    }},
    # 7: Ooty, TN
    {"location_name": "Ooty, TN", "data": {
        "latitude": 11.4100, "longitude": 76.7000, "elevation": 2240.0, "soil_ph": 5.8,
        "soil_nitrogen": 270, "soil_phosphorus": 190, "soil_potassium": 210, "soil_moisture": 420,
        "soil_cec": 260, "avg_temperature": 15.0, "min_temperature": 5.0, "avg_humidity": 80.0,
        "min_humidity": 55.0, "avg_wind_speed": 6.0, "total_rainfall": 1400.0, "historical_crops": ["Tea", "Potato", "Carrot"]
    }},
    # 8: Bhopal, MP
    {"location_name": "Bhopal, MP", "data": {
        "latitude": 23.2599, "longitude": 77.4126, "elevation": 527.0, "soil_ph": 6.9,
        "soil_nitrogen": 230, "soil_phosphorus": 160, "soil_potassium": 220, "soil_moisture": 360,
        "soil_cec": 230, "avg_temperature": 26.5, "min_temperature": 14.0, "avg_humidity": 60.0,
        "min_humidity": 28.0, "avg_wind_speed": 10.0, "total_rainfall": 1000.0, "historical_crops": ["Soybean", "Wheat", "Chickpea"]
    }},
    # 9: Visakhapatnam, AP
    {"location_name": "Visakhapatnam, AP", "data": {
        "latitude": 17.6868, "longitude": 83.2185, "elevation": 45.0, "soil_ph": 6.7,
        "soil_nitrogen": 210, "soil_phosphorus": 155, "soil_potassium": 215, "soil_moisture": 410,
        "soil_cec": 210, "avg_temperature": 29.0, "min_temperature": 22.0, "avg_humidity": 75.0,
        "min_humidity": 55.0, "avg_wind_speed": 14.0, "total_rainfall": 1100.0, "historical_crops": ["Rice", "Sugarcane", "Groundnut"]
    }},
    # 10: Dehradun, UK
    {"location_name": "Dehradun, UK", "data": {
        "latitude": 30.3165, "longitude": 78.0322, "elevation": 640.0, "soil_ph": 6.6,
        "soil_nitrogen": 250, "soil_phosphorus": 175, "soil_potassium": 225, "soil_moisture": 390,
        "soil_cec": 235, "avg_temperature": 22.0, "min_temperature": 8.0, "avg_humidity": 70.0,
        "min_humidity": 40.0, "avg_wind_speed": 5.0, "total_rainfall": 2000.0, "historical_crops": ["Basmati Rice", "Maize", "Wheat", "Lychee"]
    }}
]

# --- List to store results ---
results = []

# --- Standard Headers ---
headers = {'Content-Type': 'application/json'}

# --- Loop through payloads and send requests ---
print(f"Sending requests to {API_URL}...")
for i, payload_item in enumerate(test_payloads):
    location = payload_item["location_name"]
    data = payload_item["data"]
    print(f"\n[{i+1}/10] Testing Location: {location}...")

    try:
        # Send POST request
        response = requests.post(API_URL, headers=headers, json=data, timeout=REQUEST_TIMEOUT)

        # Check for HTTP errors (like 4xx or 5xx)
        response.raise_for_status()

        # Parse successful response
        response_data = response.json()
        results.append({
            "location": location,
            "status": "Success",
            "recommendations": response_data.get("recommendations", "N/A"),
            "processing_time": response_data.get("processing_time_seconds", -1)
        })
        print(f"  Status: Success ({response.status_code})")
        print(f"  Recommendations: {results[-1]['recommendations']}") # Print the last added result

    except requests.exceptions.Timeout:
        print(f"  Status: Error - Request timed out after {REQUEST_TIMEOUT} seconds.")
        results.append({"location": location, "status": "Error", "message": "Request Timeout"})
    except requests.exceptions.HTTPError as e:
        print(f"  Status: Error - HTTP Error ({e.response.status_code})")
        try:
            error_detail = e.response.json()
        except json.JSONDecodeError:
            error_detail = e.response.text
        results.append({"location": location, "status": "HTTP Error", "code": e.response.status_code, "message": error_detail})
    except requests.exceptions.RequestException as e:
        print(f"  Status: Error - Request failed ({type(e).__name__}): {e}")
        results.append({"location": location, "status": "Error", "message": str(e)})
    except json.JSONDecodeError:
        print(f"  Status: Error - Failed to decode JSON response.")
        results.append({"location": location, "status": "Error", "message": "Invalid JSON Response", "response_text": response.text if 'response' in locals() else 'N/A'})

    # Optional delay between requests
    time.sleep(0.5) # Wait half a second

# --- Print Summary of Results ---
print("\n" + "="*20 + " TEST SUMMARY " + "="*20)
for result in results:
    print(f"\nLocation: {result['location']}")
    print(f"Status:   {result['status']}")
    if result['status'] == 'Success':
        print(f"Recs:     {result['recommendations']}")
        print(f"Time (s): {result['processing_time']}")
    else:
        print(f"Details:  {result.get('message', 'No details')}")
print("\n" + "="*54)