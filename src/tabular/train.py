import pandas as pd
import sqlite3
import xgboost as xgb
import os
import yaml
from datetime import datetime, timedelta

def load_data():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "..", "..", "data")
    config_dir = os.path.join(base_dir, "..", "..", "config")
    
    with open(os.path.join(config_dir, "cost_matrix.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # 1. Load tabular + historical counters
    df = pd.read_csv(os.path.join(data_dir, "synthetic_data.csv"), parse_dates=["timestamp"])
    
    # 2. Join offline feature store (ring_risk_score)
    conn = sqlite3.connect(os.path.join(data_dir, "feature_store.db"))
    df_offline = pd.read_sql("SELECT user_id, ring_risk_score FROM offline_features", conn)
    conn.close()
    
    df = df.merge(df_offline, on="user_id", how="left")
    df["ring_risk_score"].fillna(0.1, inplace=True) # default
    
    # 3. Label-lag cutoff (Phase 4 requirement)
    lag_days = config["label_lag"]["exclude_last_n_days"]
    max_date = df["timestamp"].max()
    cutoff_date = max_date - timedelta(days=lag_days)
    
    # 4. Chronological split (80/20 of the labeled data)
    # Exclude recent unlabelled data from training/eval completely (for real life you'd just drop the label)
    df_labeled = df[df["timestamp"] <= cutoff_date].copy()
    
    # Chronological sort
    df_labeled = df_labeled.sort_values("timestamp")
    split_idx = int(len(df_labeled) * 0.8)
    
    train_df = df_labeled.iloc[:split_idx]
    test_df = df_labeled.iloc[split_idx:]
    
    print(f"Data ranges:")
    print(f"Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Test:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    print(f"Excluded (Label Lag): > {cutoff_date}")
    
    # Features
    features = [
        "amount", "hour_of_day", "is_weekend", "account_age_days", "historical_chargebacks",
        "device_reuse_count", "instrument_reuse_count", "velocity_5min", "velocity_1hr",
        "ring_risk_score"
    ]
    
    # Ensure sorted order for reproducibility and inference
    features.sort()
    
    X_train = train_df[features]
    y_train = train_df["label"]
    X_test = test_df[features]
    y_test = test_df["label"]
    
    # Cost sensitive weighting
    c_ltv = config["cost"]["false_positive_ltv"]
    c_cb = config["cost"]["false_negative_chargeback"]
    scale_pos_weight = c_cb / c_ltv
    
    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=10)
    
    # Save model using booster to model.json
    model_path = os.path.join(data_dir, "model.json")
    model.get_booster().save_model(model_path)
    print(f"Saved model to {model_path}")
    
    # Quick eval
    preds = model.predict_proba(X_test)[:, 1]
    
    # We will use the sweep in eval to set optimal threshold, here just log AUC
    from sklearn.metrics import roc_auc_score
    print(f"Test AUC: {roc_auc_score(y_test, preds):.4f}")

if __name__ == "__main__":
    load_data()
