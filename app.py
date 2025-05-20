import pandas as pd
from prophet import Prophet
# from prophet.diagnostics import cross_validation, performance_metrics # Kept for potential future use
from flask import Flask, request, jsonify
from io import StringIO
import logging
import requests # New import for fetching data from URL

# Suppress Prophet's informational messages during fitting
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

# --- URL for the CSV data ---
# Replace this with your actual public URL if needed
DATA_URL = "https://rgqsmlv5sa.blob.core.windows.net/cvscontainer/ModelTrainingData.csv"

# Global dictionary to store trained models
prophet_models = {}
app = Flask(__name__)

def load_and_prepare_data(data_source_url):
    """
    Fetches CSV data from a URL, loads it into a pandas DataFrame,
    converts dates, and renames columns for Prophet.
    """
    print(f"Fetching data from: {data_source_url}")
    try:
        response = requests.get(data_source_url)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from URL: {e}")
        raise  # Re-raise the exception to stop the script if data can't be loaded
    
    csv_content = response.text
    print(csv_content)
    
    # The first character might be a BOM (Byte Order Mark) if the file was saved with certain encodings
    if csv_content.startswith('\ufeff'):
        csv_content = csv_content.lstrip('\ufeff')
    
    df = pd.read_csv(StringIO(csv_content))
    df['OperationDate'] = pd.to_datetime(df['OperationDate'])
    df.rename(columns={'OperationDate': 'ds', 'Units': 'y'}, inplace=True)
    print("Data loaded and prepared successfully.")
    return df

def train_prophet_models(data_df):
    """Trains a Prophet model for each ProductId and stores it."""
    global prophet_models
    product_ids = data_df['ProductId'].unique()
    
    for pid in product_ids:
        print(f"Training model for ProductId: {pid}")
        product_df = data_df[data_df['ProductId'] == pid][['ds', 'y']].copy()
        
        if len(product_df) < 10: 
            print(f"Skipping ProductId {pid} due to insufficient data points: {len(product_df)}")
            continue
            
        model = Prophet(
            yearly_seasonality=True, 
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        try:
            model.fit(product_df)
            prophet_models[pid] = model
            print(f"Model for ProductId: {pid} trained successfully.")
        except Exception as e:
            print(f"Error training model for ProductId {pid}: {e}")
            
    print(f"Trained {len(prophet_models)} models.")

@app.route('/predict', methods=['GET'])
def predict():
    """
    API endpoint to predict units for a given product_id and future_date.
    Query parameters:
        - product_id (int): The ID of the product.
        - future_date (str): The date for prediction (YYYY-MM-DD).
    """
    product_id_str = request.args.get('product_id')
    future_date_str = request.args.get('future_date')

    if not product_id_str or not future_date_str:
        return jsonify({"error": "Missing product_id or future_date parameter"}), 400

    try:
        product_id = int(product_id_str)
    except ValueError:
        return jsonify({"error": "product_id must be an integer"}), 400

    try:
        future_date = pd.to_datetime(future_date_str)
    except ValueError:
        return jsonify({"error": "future_date must be in YYYY-MM-DD format"}), 400

    if product_id not in prophet_models:
        return jsonify({"error": f"No model found for product_id {product_id}. Available IDs: {list(prophet_models.keys())}"}), 404

    model = prophet_models[product_id]
    future_df = pd.DataFrame({'ds': [future_date]})
    
    try:
        forecast = model.predict(future_df)
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

    predicted_units = forecast['yhat'].iloc[0]
    predicted_units_adjusted = max(0, round(predicted_units))
    yhat_lower = max(0, round(forecast['yhat_lower'].iloc[0]))
    yhat_upper = max(0, round(forecast['yhat_upper'].iloc[0]))

    return jsonify({
        "product_id": product_id,
        "date": future_date_str,
        "predicted_units": predicted_units_adjusted,
        "predicted_units_raw": float(predicted_units),
        "prediction_interval_lower": yhat_lower,
        "prediction_interval_upper": yhat_upper
    })

@app.route('/status', methods=['GET'])
def status():
    """Simple status endpoint to check if the app is running and models are loaded."""
    return jsonify({
        "status": "API is running",
        "data_source_url": DATA_URL,
        "trained_models_for_product_ids": list(prophet_models.keys()),
        "total_models_trained": len(prophet_models)
    })

if __name__ == '__main__':
    print("Loading data and training models...")
    try:
        main_df = load_and_prepare_data(DATA_URL)
        train_prophet_models(main_df)
        
        # Example of how to use the model directly (without API)
        if 9 in prophet_models:
            sample_model = prophet_models[9]
            sample_future_date = pd.to_datetime('2024-01-01')
            sample_future_df = pd.DataFrame({'ds': [sample_future_date]})
            sample_forecast = sample_model.predict(sample_future_df)
            print("\nSample direct prediction for ProductId 9 on 2024-01-01:")
            print(sample_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

        if 35 in prophet_models:
            sample_model = prophet_models[35]
            sample_future_date = pd.to_datetime('2024-06-15')
            sample_future_df = pd.DataFrame({'ds': [sample_future_date]})
            sample_forecast = sample_model.predict(sample_future_df)
            print("\nSample direct prediction for ProductId 35 on 2024-06-15:")
            print(sample_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

        print("\nStarting Flask server...")
        print("API Endpoints:")
        print("  GET /status")
        print("  GET /predict?product_id=<id>&future_date=<YYYY-MM-DD>")
        print("Example: http://127.0.0.1:5000/predict?product_id=9&future_date=2024-01-01")
        
        app.run(host='0.0.0.0', port=5000, debug=True)

    except Exception as e:
        print(f"Failed to initialize or start the application: {e}")