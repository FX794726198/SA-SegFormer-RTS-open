#!/usr/bin/env python
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "scripts"))
from train_baseline import main

if __name__ == "__main__":
    main(default_model="convnext")
