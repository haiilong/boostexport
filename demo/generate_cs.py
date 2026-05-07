"""
Regenerate all three C# model files into ModelTests/.
Run from the demo/ directory:
    python generate_cs.py
"""

import subprocess
import sys

exporter = "../export_model.py"

models = [
    ("model_lgb.txt",  "LgbmModel", "DotnetModelTests/LgbmModel.cs"),
    ("model_xgb.json", "XgbModel",  "DotnetModelTests/XgbModel.cs"),
    ("model_cb.cbm",   "CbModel",   "DotnetModelTests/CbModel.cs"),
]

for model_path, class_name, output in models:
    print(f"Exporting {model_path} -> {output}")
    subprocess.run(
        [sys.executable, exporter, model_path, "-c", class_name, "-o", output],
        check=True,
    )
    print()
