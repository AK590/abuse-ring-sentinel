import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import sqlite3

def generate_synthetic_data(num_users=1000, num_txs=10000, days=100):
    np.random.seed(42)
    random.seed(42)

    # Users
    users = [f"u_{i}" for i in range(num_users)]
    
    # Devices & Instruments (Normally 1-to-few per user)
    devices = [f"d_{i}" for i in range(num_users)]
    instruments = [f"inst_{i}" for i in range(num_users)]
    
    # Inject fraud rings (e.g. 50 users sharing 5 devices and 5 instruments)
    fraud_users = users[-50:]
    fraud_devices = [f"d_fraud_{i}" for i in range(5)]
    fraud_instruments = [f"inst_fraud_{i}" for i in range(5)]
    
    start_date = datetime.now() - timedelta(days=days)
    
    data = []
    
    for i in range(num_txs):
        # 5% transactions are from fraud ring
        is_fraud = random.random() < 0.05
        if is_fraud:
            u = random.choice(fraud_users)
            d = random.choice(fraud_devices)
            inst = random.choice(fraud_instruments)
            amount = np.random.uniform(10, 50) # Fraudsters testing small amounts
            is_weekend = 1 if (start_date.weekday() >= 5) else 0
        else:
            u = random.choice(users[:-50])
            idx = users.index(u)
            d = devices[idx]
            inst = instruments[idx]
            amount = np.random.uniform(50, 1000)
        
        # Calculate random timestamp within the 'days' period
        days_offset = random.uniform(0, days)
        tx_time = start_date + timedelta(days=days_offset)
        
        # Features
        hour = tx_time.hour
        is_weekend = 1 if (tx_time.weekday() >= 5) else 0
        account_age = random.uniform(10, 365)
        
        # We simulate some lag label: if tx was recent (last 60 days), it's "pending/unknown"
        # For simplicity in this dataset, we just output the true label, and let label_lag.py mask it.
        label = 1 if is_fraud else 0
        
        data.append({
            "transaction_id": f"tx_{i}",
            "user_id": u,
            "device_hash": d,
            "instrument_id": inst,
            "amount": amount,
            "timestamp": tx_time,
            "hour_of_day": hour,
            "is_weekend": is_weekend,
            "account_age_days": account_age,
            "historical_chargebacks": 1 if (is_fraud and random.random() < 0.2) else 0,
            # We mock real-time counters historically for training, in real life you'd pull from a historic log
            # For simplicity, we just inject noisy proxies based on fraud vs non-fraud
            "device_reuse_count": random.randint(3, 10) if is_fraud else 1,
            "instrument_reuse_count": random.randint(3, 10) if is_fraud else 1,
            "velocity_5min": random.randint(2, 5) if is_fraud else random.randint(0, 1),
            "velocity_1hr": random.randint(5, 10) if is_fraud else random.randint(0, 2),
            "label": label
        })
        
    df = pd.DataFrame(data)
    df = df.sort_values("timestamp")
    
    # Save to CSV
    os.makedirs(os.path.join(os.path.dirname(__file__)), exist_ok=True)
    csv_path = os.path.join(os.path.dirname(__file__), "synthetic_data.csv")
    df.to_csv(csv_path, index=False)
    
    # Generate SQLite mock for Feature Store
    db_path = os.path.join(os.path.dirname(__file__), "feature_store.db")
    conn = sqlite3.connect(db_path)
    
    # Mock some ring_risk_score offline features
    offline_feats = []
    for u in users:
        is_fraud = u in fraud_users
        # R-GCN would find this
        score = random.uniform(0.7, 1.0) if is_fraud else random.uniform(0.0, 0.3)
        offline_feats.append({"user_id": u, "ring_risk_score": score})
        
    df_offline = pd.DataFrame(offline_feats)
    df_offline.to_sql("offline_features", conn, if_exists="replace", index=False)
    conn.close()

if __name__ == "__main__":
    generate_synthetic_data()
    print("Generated synthetic_data.csv and feature_store.db")
