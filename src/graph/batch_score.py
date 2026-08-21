import pandas as pd
import sqlite3
import os
import time
import torch
from src.graph.schema import build_heterograph
from src.graph.model import RGCN

def run_batch_job():
    start_time = time.perf_counter()
    
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "..", "..", "data")
    
    # 1. Load data
    df = pd.read_csv(os.path.join(data_dir, "synthetic_data.csv"))
    
    # 2. Build graph
    g, users = build_heterograph(df)
    
    # 3. Model
    model = RGCN(in_feats=10, hid_feats=16, out_feats=8, rel_names=g.etypes)
    
    # In a real scenario, we'd load pre-trained weights here
    # model.load_state_dict(torch.load("rgcn_weights.pt"))
    model.eval()
    
    with torch.no_grad():
        inputs = {ntype: g.nodes[ntype].data['feat'] for ntype in g.ntypes}
        scores = model(g, inputs).squeeze().numpy()
        
    # 4. Write to Feature Store (SQLite mock of Postgres)
    offline_feats = pd.DataFrame({
        "user_id": users,
        "ring_risk_score": scores
    })
    
    db_path = os.path.join(data_dir, "feature_store.db")
    conn = sqlite3.connect(db_path)
    offline_feats.to_sql("offline_features", conn, if_exists="replace", index=False)
    
    # Add a metadata table for monitoring
    meta = pd.DataFrame([{"last_run": time.time()}])
    meta.to_sql("batch_metadata", conn, if_exists="replace", index=False)
    
    conn.close()
    
    end_time = time.perf_counter()
    print(f"Batch job completed in {end_time - start_time:.2f} seconds. Scored {len(users)} users.")

if __name__ == "__main__":
    run_batch_job()
