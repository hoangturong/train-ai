import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Show only ERROR and above

import requests
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import check_array
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from flask import Flask, request, jsonify
import threading
import time
from datetime import datetime
import pickle
import logging
import itertools
from scipy.stats import entropy, skew
from typing import List, Tuple, Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
BASE_URL = "http://localhost:3001"
DB_NAME = "taixiu_data.db"
WINDOW_SIZE = 10
MODEL_FILE = "best_models.pkl"

# Flask app
app = Flask(__name__)

# Database initialization
def init_db() -> None:
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS results 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                     total INTEGER, 
                     data TEXT, 
                     type TEXT, 
                     timestamp TEXT)''')
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Database initialization error: {e}")
    finally:
        conn.close()

# Fetch data from API
def fetch_history_from_api(num_results: int = 1000) -> List[Dict]:
    try:
        response = requests.get(f"{BASE_URL}/data/{num_results}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Error fetching history: {e}")
        return []

def generate_data_from_api(num_sessions: int = 200, rolls_per_session: int = 3) -> List[Dict]:
    results = []
    try:
        for _ in range(num_sessions):
            response = requests.get(f"{BASE_URL}/game/{rolls_per_session}", timeout=10)
            response.raise_for_status()
            results.append(response.json())
        return results
    except requests.RequestException as e:
        logging.error(f"Error generating data: {e}")
        return []

# Save to database
def save_to_db(data: List[Dict]) -> None:
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        for item in data:
            total = item.get("response", 0) if isinstance(item, dict) else item
            data_str = ",".join(map(str, item.get("data", []))) if isinstance(item, dict) else ""
            result_type = item.get("type", "Unknown") if isinstance(item, dict) else "Unknown"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("INSERT INTO results (total, data, type, timestamp) VALUES (?, ?, ?, ?)",
                     (total, data_str, result_type, timestamp))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Database save error: {e}")
    finally:
        conn.close()

# Load data from database
def load_all_data_from_db() -> List[Tuple[int, str]]:
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT total, timestamp FROM results ORDER BY timestamp ASC")
        data = c.fetchall()
        return [(row[0], row[1]) for row in data]
    except sqlite3.Error as e:
        logging.error(f"Database load error: {e}")
        return []
    finally:
        conn.close()

# Statistics calculation
def get_stats(totals: List[Tuple[int, str]]) -> Dict[str, float]:
    if not totals:
        return {"total_records": 0, "average_total": 0.0, "tai_ratio": 0.5, "streak": 0}
    totals_only = [t[0] for t in totals]
    tai_count = sum(1 for t in totals_only[-50:] if t >= 11)
    streak = max((sum(1 for _ in g) for k, g in itertools.groupby(totals_only[-20:], lambda x: x >= 11)), default=0)
    return {
        "total_records": len(totals),
        "average_total": float(np.mean(totals_only)),
        "tai_ratio": tai_count / min(50, len(totals)),
        "streak": streak
    }

# Feature engineering
def prepare_features(totals_with_time: List[Tuple[int, str]], window_size: int = WINDOW_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    totals = [t[0] for t in totals_with_time]
    timestamps = [datetime.strptime(t[1], "%Y-%m-%d %H:%M:%S") for t in totals_with_time]
    X, y = [], []
    for i in range(len(totals) - window_size):
        window = totals[i:i + window_size]
        next_result = totals[i + window_size]
        tai_ratio = sum(1 for x in window if x >= 11) / window_size
        std_dev = np.std(window)
        streak = max((sum(1 for _ in g) for k, g in itertools.groupby(window, lambda x: x >= 11)), default=0)
        moving_avg = np.mean(window)
        hist, _ = np.histogram(window, bins=16, range=(3, 18))
        window_entropy = entropy(hist + 1e-10)
        skewness = skew(window)
        time_diffs = [(timestamps[i + j + 1] - timestamps[i + j]).total_seconds() for j in range(window_size - 1)]
        avg_time_diff = np.mean(time_diffs) if time_diffs else 0
        X.append(list(window) + [tai_ratio, std_dev, streak, moving_avg, window_entropy, skewness, avg_time_diff])
        y.append(1 if next_result >= 11 else 0)
    return np.array(X), np.array(y)

# Build LSTM model
def build_lstm_model(window_size: int) -> Sequential:
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train stacking model
def train_stacking(totals_with_time: List[Tuple[int, str]], window_size: int = WINDOW_SIZE) -> Tuple[Any, Any, Any, Any, Any]:
    try:
        X, y = prepare_features(totals_with_time, window_size)
        if len(X) == 0:
            raise ValueError("Insufficient data for training")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        feature_names = [f'roll_{i}' for i in range(window_size)] + [
            'tai_ratio', 'std_dev', 'streak', 'moving_avg', 'window_entropy', 'skewness', 'avg_time_diff'
        ]

        # Gradient Boosting
        gb_params = {'n_estimators': [200, 300], 'learning_rate': [0.01, 0.05], 'max_depth': [3, 5]}
        gb_model = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            gb_params, cv=5, n_jobs=-1
        ).fit(X_train, y_train).best_estimator_

        # XGBoost
        xgb_params = {'n_estimators': [200, 300], 'learning_rate': [0.01, 0.05], 'max_depth': [3, 5]}
        xgb_model = GridSearchCV(
            XGBClassifier(random_state=42, eval_metric='logloss'),
            xgb_params, cv=5, n_jobs=-1
        ).fit(X_train, y_train).best_estimator_

        # LightGBM
        lgb_params = {
            'n_estimators': [200, 300],
            'learning_rate': [0.01, 0.05],
            'max_depth': [3, 5],
            'min_split_gain': [0.1],
            'min_child_samples': [20],
            'verbose': [-1]
        }
        lgb_model = GridSearchCV(
            LGBMClassifier(random_state=42),
            lgb_params, cv=5, n_jobs=-1
        ).fit(X_train, y_train).best_estimator_


        # LSTM
        X_train_lstm = X_train[:, :window_size].reshape((X_train.shape[0], window_size, 1))
        X_test_lstm = X_test[:, :window_size].reshape((X_test.shape[0], window_size, 1))
        lstm_model = build_lstm_model(window_size)
        lstm_model.fit(
            X_train_lstm, y_train,
            epochs=30, batch_size=32, verbose=0,
            callbacks=[EarlyStopping(monitor='loss', patience=5)]
        )

        # Meta predictions
        meta_train = np.column_stack((
            gb_model.predict_proba(X_train)[:, 1],
            xgb_model.predict_proba(X_train)[:, 1],
            lgb_model.predict_proba(X_train)[:, 1],
            lstm_model.predict(X_train_lstm, verbose=0).flatten()
        ))
        meta_test = np.column_stack((
            gb_model.predict_proba(X_test)[:, 1],
            xgb_model.predict_proba(X_test)[:, 1],
            lgb_model.predict_proba(X_test)[:, 1],
            lstm_model.predict(X_test_lstm, verbose=0).flatten()
        ))

        # Meta model
        meta_model = XGBClassifier(random_state=42, eval_metric='logloss').fit(meta_train, y_train)
        accuracy = (meta_model.predict(meta_test) == y_test).mean()
        logging.info(f"Enhanced Stacking Test Accuracy: {accuracy * 100:.2f}%")
        return gb_model, xgb_model, lgb_model, lstm_model, meta_model
    except Exception as e:
        logging.error(f"Training error: {e}")
        return None, None, None, None, None

# Prediction function
def predict_stacking(gb_model, xgb_model, lgb_model, lstm_model, meta_model, session: List[int], window_size: int = WINDOW_SIZE) -> Tuple[Optional[str], float]:
    try:
        if len(session) < window_size or not all(model is not None for model in [gb_model, xgb_model, lgb_model, lstm_model, meta_model]):
            return None, 0.0
        
        input_data = np.array(session[-window_size:], dtype=float)
        tai_ratio = sum(1 for x in input_data if x >= 11) / window_size
        std_dev = np.std(input_data)
        streak = max((sum(1 for _ in g) for k, g in itertools.groupby(input_data, lambda x: x >= 11)), default=0)
        moving_avg = np.mean(input_data)
        hist, _ = np.histogram(input_data, bins=16, range=(3, 18))
        window_entropy = entropy(hist + 1e-10)
        skewness = skew(input_data)
        avg_time_diff = 0.0
        
        features = np.append(input_data, [tai_ratio, std_dev, streak, moving_avg, window_entropy, skewness, avg_time_diff]).reshape(1, -1)
        features = check_array(features, ensure_2d=True)

        gb_pred = gb_model.predict_proba(features)[:, 1]
        xgb_pred = xgb_model.predict_proba(features)[:, 1]
        lgb_pred = lgb_model.predict_proba(features)[:, 1]
        lstm_input = features[:, :window_size].reshape((1, window_size, 1))
        lstm_pred = lstm_model.predict(lstm_input, verbose=0).flatten()
        meta_input = np.column_stack((gb_pred, xgb_pred, lgb_pred, lstm_pred))
        
        final_pred = meta_model.predict(meta_input)[0]
        confidence = meta_model.predict_proba(meta_input)[0].max()
        return "Tài" if final_pred == 1 else "Xỉu", float(confidence)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return None, 0.0

# Global variables
model_ref = [None, None, None, None, None]  #
#[gb, xgb, lgb, lstm, meta]
best_accuracy = 0.0
training_active = True
feedback_data = []

# Auto training
def auto_train() -> None:
    global model_ref, best_accuracy
    while True:
        if not training_active:
            time.sleep(5)
            continue
        try:
            logging.info("Fetching new data...")
            new_data = generate_data_from_api(500)
            if new_data:
                save_to_db(new_data)
                logging.info(f"Added {len(new_data)} new results.")
            
            totals = load_all_data_from_db()
            if len(totals) >= WINDOW_SIZE + 1:
                logging.info("Training enhanced stacking model...")
                models = train_stacking(totals, WINDOW_SIZE)
                if all(models):
                    gb, xgb, lgb, lstm, meta = models
                    X, y = prepare_features(totals, WINDOW_SIZE)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    meta_test = np.column_stack((
                        gb.predict_proba(X_test)[:, 1],
                        xgb.predict_proba(X_test)[:, 1],
                        lgb.predict_proba(X_test)[:, 1],
                        lstm.predict(X_test[:, :WINDOW_SIZE].reshape(-1, WINDOW_SIZE, 1), verbose=0).flatten()
                    ))
                    accuracy = (meta.predict(meta_test) == y_test).mean()
                    if accuracy > best_accuracy:
                        model_ref = [gb, xgb, lgb, lstm, meta]
                        best_accuracy = accuracy
                        logging.info(f"Updated best model: {best_accuracy * 100:.2f}%")
                        with open(MODEL_FILE, "wb") as f:
                            pickle.dump(model_ref, f)
        except Exception as e:
            logging.error(f"Auto-train error: {e}")
        time.sleep(15)

# API endpoints
@app.route('/predict', methods=['POST'])
def predict() -> jsonify:
    try:
        if not all(model_ref):
            return jsonify({"error": "Models not ready"}), 503
        data = request.json
        session = data.get("session", [])
        prediction, confidence = predict_stacking(*model_ref, session, WINDOW_SIZE)
        if prediction is None:
            return jsonify({"error": f"Need at least {WINDOW_SIZE} results"}), 400
        stats = get_stats(load_all_data_from_db())
        return jsonify({"prediction": prediction, "confidence": confidence, "stats": stats})
    except Exception as e:
        logging.error(f"Predict endpoint error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/control', methods=['POST'])
def control() -> jsonify:
    global training_active
    try:
        action = request.json.get("action")
        if action == "start":
            training_active = True
            return jsonify({"message": "Training started"})
        elif action == "stop":
            training_active = False
            return jsonify({"message": "Training stopped"})
        elif action == "stats":
            return jsonify(get_stats(load_all_data_from_db()))
        return jsonify({"error": "Invalid action"}), 400
    except Exception as e:
        logging.error(f"Control endpoint error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/feedback', methods=['POST'])
def feedback() -> jsonify:
    try:
        data = request.json
        actual_result = data.get("actual_result")
        if actual_result is not None:
            feedback_data.append({"total": actual_result, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            save_to_db([{"response": actual_result}])
            logging.info(f"Received feedback: {actual_result}")
            return jsonify({"message": "Feedback recorded"})
        return jsonify({"error": "Invalid feedback"}), 400
    except Exception as e:
        logging.error(f"Feedback endpoint error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Main function
def main() -> None:
    logging.info("Starting enhanced server...")
    init_db()
    
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                model_ref[:] = pickle.load(f)
                logging.info("Loaded pre-trained models.")
        except Exception as e:
            logging.error(f"Model loading error: {e}")
    
    history_data = fetch_history_from_api(2000)
    if history_data:
        save_to_db(history_data)
        logging.info(f"Saved {len(history_data)} history results.")
    
    totals = load_all_data_from_db()
    if len(totals) >= WINDOW_SIZE + 1 and not all(model_ref):
        model_ref[:] = train_stacking(totals, WINDOW_SIZE)
    
    train_thread = threading.Thread(target=auto_train, daemon=True)
    train_thread.start()
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)

if __name__ == "__main__":
    main()