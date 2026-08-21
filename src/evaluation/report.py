import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
import os
import yaml

def generate_report():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "..", "..", "data")
    config_dir = os.path.join(base_dir, "..", "..", "config")
    
    with open(os.path.join(config_dir, "cost_matrix.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # Load model and test set (for simplicity we just re-split or use the generated test set)
    # Re-doing the split to match train.py exactly
    df = pd.read_csv(os.path.join(data_dir, "synthetic_data.csv"), parse_dates=["timestamp"])
    
    import sqlite3
    conn = sqlite3.connect(os.path.join(data_dir, "feature_store.db"))
    df_offline = pd.read_sql("SELECT user_id, ring_risk_score FROM offline_features", conn)
    conn.close()
    
    df = df.merge(df_offline, on="user_id", how="left")
    df["ring_risk_score"].fillna(0.1, inplace=True)
    
    lag_days = config["label_lag"]["exclude_last_n_days"]
    max_date = df["timestamp"].max()
    cutoff_date = max_date - pd.Timedelta(days=lag_days)
    
    df_labeled = df[df["timestamp"] <= cutoff_date].copy().sort_values("timestamp")
    split_idx = int(len(df_labeled) * 0.8)
    test_df = df_labeled.iloc[split_idx:]
    
    features = [
        "amount", "hour_of_day", "is_weekend", "account_age_days", "historical_chargebacks",
        "device_reuse_count", "instrument_reuse_count", "velocity_5min", "velocity_1hr",
        "ring_risk_score"
    ]
    features.sort()
    
    X_test = test_df[features]
    y_test = test_df["label"]
    
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(data_dir, "model.json"))
    
    preds_proba = model.predict_proba(X_test)[:, 1]
    
    threshold = config["thresholds"]["challenge"]
    preds = (preds_proba >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    fpr = fp / (fp + tn)
    
    c_ltv = config["cost"]["false_positive_ltv"]
    c_cb = config["cost"]["false_negative_chargeback"]
    total_cost = (fp * c_ltv) + (fn * c_cb)
    
    print("=== Evaluation Report ===")
    print(f"Test Set Size: {len(X_test)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"Total Business Cost: ${total_cost:.2f}")
    print("=========================")

if __name__ == "__main__":
    generate_report()
