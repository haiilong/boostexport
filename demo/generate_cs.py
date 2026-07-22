"""
Regenerate all C# model files into DotnetModelTests/.
Run from the demo/ directory:
    python generate_cs.py
"""

import subprocess
import sys

exporter = "../export_model.py"

models = [
    # regression
    ("model_lgb.txt",        "LgbmModel",      "DotnetModelTests/LgbmModel.cs"),
    ("model_xgb.json",       "XgbModel",       "DotnetModelTests/XgbModel.cs"),
    ("model_cb.cbm",         "CbModel",        "DotnetModelTests/CbModel.cs"),
    # binary classification
    ("model_lgb_bin.txt",    "LgbmBinModel",   "DotnetModelTests/LgbmBinModel.cs"),
    ("model_xgb_bin.json",   "XgbBinModel",    "DotnetModelTests/XgbBinModel.cs"),
    ("model_cb_bin.cbm",     "CbBinModel",     "DotnetModelTests/CbBinModel.cs"),
    # multiclass classification
    ("model_lgb_multi.txt",  "LgbmMultiModel", "DotnetModelTests/LgbmMultiModel.cs"),
    ("model_xgb_multi.json", "XgbMultiModel",  "DotnetModelTests/XgbMultiModel.cs"),
    ("model_cb_multi.cbm",   "CbMultiModel",   "DotnetModelTests/CbMultiModel.cs"),
    # regression with missing values
    ("model_lgb_nan.txt",    "LgbmNanModel",   "DotnetModelTests/LgbmNanModel.cs"),
    ("model_xgb_nan.json",   "XgbNanModel",    "DotnetModelTests/XgbNanModel.cs"),
    ("model_cb_nan.cbm",     "CbNanModel",     "DotnetModelTests/CbNanModel.cs"),
]

for model_path, class_name, output in models:
    print(f"Exporting {model_path} -> {output}")
    subprocess.run(
        [sys.executable, exporter, model_path, "-c", class_name, "-o", output],
        check=True,
    )
    print()
