"""Rebuild models and evaluation tables without opening a notebook."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ev_project.modeling import run_training

if __name__ == '__main__':
    run_training(ROOT)
