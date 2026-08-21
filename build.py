"""Nixpacks build hook — runs during Railway build to generate data & model."""
import subprocess
import sys

scripts = [
    [sys.executable, "data/synthetic_generator.py"],
    [sys.executable, "src/tabular/train.py"],
    [sys.executable, "src/graph/batch_score.py"],
]

for cmd in scripts:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Done.\n")
