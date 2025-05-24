import os
import pandas as pd
from prophet import Prophet
# from prophet.diagnostics import cross_validation, performance_metrics # Kept for potential future use
from flask import Flask, request, jsonify
app = Flask(__name__)
from io import StringIO
import logging
import requests # New import for fetching data from URL
import threading
from datetime import datetime
import time # Only for potential simulation if needed, not strictly for this logic

logging.basicConfig(filename='logs\\app.log', level=logging.INFO)

# --- Global Variables ---
MODELS = {}
DATA_URL = "https://rgqsmlv5sa.blob.core.windows.net/cvscontainer/ModelTrainingData.csv"
MIN_DATA_POINTS_FOR_TRAINING = 5 # You can adjust this threshold

# --- State for Asynchronous Training ---
training_lock = threading.Lock()
_is_training_active = False
_last_training_status = "No training has been initiated yet."
_last_successful_training_time = None
_models_trained_in_last_run = 0

# --- Flask App Initialization ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- Helper Functions ---
def preprocess_data(df):
    """Prepares the DataFrame for Prophet."""
    # Handle potential BOM in the first column name
    if df.columns[0].startswith('\ufeff'):
        df.rename(columns={df.columns[0]: df.columns[0].replace('\ufeff', '')}, inplace=True)
    
    # Prophet expects columns 'ds' (datestamp) and 'y' (numeric value)
    df = df.rename(columns={'OperationDate': 'ds', 'Units': 'y', 'ProductId': 'product_id'})
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    # Ensure 'y' is numeric, coercing errors to NaN (which Prophet can handle by ignoring)
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df.dropna(subset=['ds', 'y'], inplace=True) # Drop rows where ds or y is NaN after coercion
    return df

# Context manager to suppress Prophet's verbose fitting logs
class SuppressProphetLogs:
    def __enter__(self):
        self.prophet_logger = logging.getLogger('prophet')
        self.cmdstanpy_logger = logging.getLogger('cmdstanpy')
        self.original_prophet_level = self.prophet_logger.level
        self.original_cmdstanpy_level = self.cmdstanpy_logger.level
        self.prophet_logger.setLevel(logging.WARNING)
        self.cmdstanpy_logger.setLevel(logging.WARNING)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.prophet_logger.setLevel(self.original_prophet_level)
        self.cmdstanpy_logger.setLevel(self.original_cmdstanpy_level)

def _perform_actual_training_logic():
    """
    Core logic for loading data and training models.
    This function is intended to be run in a background thread.
    It updates global MODELS.
    """
    global MODELS, _last_successful_training_time, _models_trained_in_last_run
    
    logging.info(f"Background Training: Attempting to load data from: {DATA_URL}")
    
    temp_models = {} # Train into a temporary dict
    models_trained_count_this_run = 0

    response = requests.get(DATA_URL)
    response.raise_for_status() # Raise an exception for HTTP errors
    
    csv_content = response.content.decode('utf-8-sig') # utf-8-sig handles BOM
    df_global = pd.read_csv(StringIO(csv_content))

    logging.info("Background Training: Data loaded. Preprocessing...")
    df_global = preprocess_data(df_global)
    
    # Ensure correct column name if BOM was present and not handled by utf-8-sig alone
    if 'ï»¿ProductId' in df_global.columns and 'product_id' not in df_global.columns:
        logging.info("Background Training: Renaming ProductId column due to BOM.")
        df_global.rename(columns={'ï»¿ProductId': 'product_id'}, inplace=True)

    product_ids = df_global['product_id'].unique()
    total_products_to_attempt = len(product_ids)
    logging.info(f"Background Training: Data preprocessed. Found {total_products_to_attempt} unique ProductIds.")
    
    for i, product_id in enumerate(product_ids):
        df_product = df_global[df_global['product_id'] == product_id][['ds', 'y']].copy()
        df_product.sort_values('ds', inplace=True)

        if len(df_product) < MIN_DATA_POINTS_FOR_TRAINING:
            # logging.warning(f"Background Training: Skipping ProductId '{product_id}' ({i+1}/{total_products_to_attempt}): Insufficient data points ({len(df_product)}). Needs at least {MIN_DATA_POINTS_FOR_TRAINING}.")
            continue
        
        # logging.info(f"Background Training: Training model for ProductId: {product_id} ({i+1}/{total_products_to_attempt})...") # Can be too verbose
        try:
            with SuppressProphetLogs():
                model = Prophet()
                model.fit(df_product)
            temp_models[product_id] = model
            models_trained_count_this_run += 1
        except Exception as e:
            logging.error(f"Background Training: Error training model for ProductId {product_id}: {e}")
            # Continue to the next product if one fails

    MODELS = temp_models # Atomically update the global models dictionary
    _last_successful_training_time = datetime.now().isoformat()
    _models_trained_in_last_run = models_trained_count_this_run
    logging.info(f"Background Training: Completed. Successfully trained {models_trained_count_this_run} models.")


def training_worker():
    """Worker function for the background training thread."""
    global _is_training_active, _last_training_status, _models_trained_in_last_run
    
    logging.info("Training worker thread started.")
    try:
        _perform_actual_training_logic()
        _last_training_status = (
            f"Training completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"with {_models_trained_in_last_run} models trained."
        )
        logging.info(_last_training_status)
    except Exception as e:
        logging.error(f"Training worker encountered an error: {e}", exc_info=True)
        _last_training_status = f"Training failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {str(e)}"
        global MODELS
        MODELS = {} # Clear models on failure
        _models_trained_in_last_run = 0
    finally:
        _is_training_active = False
        logging.info("Training worker thread finished.")

# --- Flask Routes ---
@app.route('/train', methods=['GET'])
def train_models_endpoint():
    global _is_training_active, _last_training_status
    
    with training_lock:
        if _is_training_active:
            message_to_return = "training is still in progress"
            details = f"A training job is currently active. Last update: {_last_training_status}"
            logging.info(f"/train called: {message_to_return}")
            return jsonify({"status": message_to_return, "details": details}), 202
        else:
            # Not currently active, so we will start a new one.
            # The message "training completed" or "training failed" refers to the *previous* completed run.
            if "completed successfully" in _last_training_status:
                message_to_return = "training completed"
                details = f"Previous run: {_last_training_status} A new training cycle is now being initiated."
            elif "failed" in _last_training_status:
                message_to_return = "training failed"
                details = f"Previous run: {_last_training_status} Attempting to start a new training cycle."
            else: # e.g., "No training has been initiated yet."
                message_to_return = "training started"
                details = "New training process initiated."

            _is_training_active = True
            _last_training_status = "Training process has been initiated..." # Initial status for this new run
            
            logging.info(f"/train called: {details}")
            
            thread = threading.Thread(target=training_worker, name="ModelTrainingThread")
            thread.daemon = True  # Allows main program to exit even if threads are running
            thread.start()
            
            return jsonify({"status": message_to_return, "details": details}), 202

@app.route('/training_status', methods=['GET'])
def get_training_status():
    """Provides the current status of the training system."""
    return jsonify({
        "is_training_active": _is_training_active,
        "status_message": _last_training_status,
        "models_currently_loaded": len(MODELS),
        "last_successful_training_time": _last_successful_training_time,
        "models_trained_in_last_successful_run": _models_trained_in_last_run if "completed successfully" in _last_training_status else 0
    }), 200

@app.route('/predict', methods=['GET'])
def predict():
    product_id = request.args.get('product_id')
    operation_date_str = request.args.get('operation_date')

    if not product_id:
        return jsonify({"error": "ProductId parameter is required."}), 400
    if not operation_date_str:
        return jsonify({"error": "OperationDate parameter is required."}), 400

    try:
        future_date = pd.to_datetime(operation_date_str)
    except ValueError:
        return jsonify({"error": "Invalid OperationDate format. Please use YYYY-MM-DD."}), 400

    if not MODELS: # Check if MODELS dict is empty (e.g., after a failed training or before first training)
        logging.warning(f"Prediction attempt for {product_id} but no models are loaded. Last training status: {_last_training_status}")
        return jsonify({"error": "Models are not available. Please trigger /train or check /training_status.", 
                        "last_training_status": _last_training_status}), 503 # Service Unavailable

    if product_id not in MODELS:
        logging.warning(f"Model for ProductId '{product_id}' not found. Available models: {len(MODELS)}")
        return jsonify({"error": f"Model for ProductId '{product_id}' not found. It might have had insufficient data or training failed for this specific product."}), 404

    model = MODELS[product_id]
    future_df = pd.DataFrame({'ds': [future_date]})
    
    try:
        logging.info(f"Predicting for ProductId: {product_id} on Date: {operation_date_str}")
        forecast = model.predict(future_df)
        
        predicted_value = forecast['yhat'].iloc[0]
        yhat_lower = forecast['yhat_lower'].iloc[0]
        yhat_upper = forecast['yhat_upper'].iloc[0]

        # Optionally, ensure predictions are non-negative
        # predicted_value = max(0, predicted_value)
        # yhat_lower = max(0, yhat_lower)
        # yhat_upper = max(0, yhat_upper)

        return jsonify({
            "product_id": product_id,
            "operation_date": operation_date_str,
            "predicted_units": round(predicted_value, 2),
            "yhat_lower": round(yhat_lower, 2),
            "yhat_upper": round(yhat_upper, 2)
        })
    except Exception as e:
        logging.error(f"Error during prediction for ProductId {product_id}: {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed for ProductId {product_id}."}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "models_loaded": len(MODELS), "last_training_status": _last_training_status}), 200

# --- Main Execution ---
if __name__ == '__main__':
    try:
        #http://127.0.0.1:5000/predict?product_id=AAIN12-AU-1&operation_date=2026-01-01

        logging.info("\nStarting Flask server...")
        logging.info("Trigger /train to begin model training, or GET /training_status to check status.")        
        logging.info("API Endpoints:")
        logging.info("  GET /health")
        logging.info("  GET /train")
        logging.info("  GET /training_status")
        logging.info("  GET /predict?product_id=<id>&future_date=<YYYY-MM-DD>")
        logging.info("Example: http://127.0.0.1:5000/predict?product_id=AAIN12-AU-1&operation_date=2026-01-01")        
        app.run(host='0.0.0.0', port=os.environ.get('WEBSITES_PORT', 5000))

    except Exception as e:
        logging.info(f"Failed to initialize or start the application: {e}")    