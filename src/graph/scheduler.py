from apscheduler.schedulers.blocking import BlockingScheduler
from src.graph.batch_score import run_batch_job
import yaml
import os

def start_scheduler():
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "cost_matrix.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    hours = config["batch_job"]["ring_score_refresh_hours"]
    
    scheduler = BlockingScheduler()
    scheduler.add_job(run_batch_job, 'interval', hours=hours)
    
    print(f"Starting scheduler to run R-GCN batch job every {hours} hours.")
    # Run once immediately
    run_batch_job()
    scheduler.start()

if __name__ == "__main__":
    start_scheduler()
