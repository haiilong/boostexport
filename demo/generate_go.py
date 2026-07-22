"""
Regenerate all Go model files into GoModelTests/.
Run from the demo/ directory:
    python generate_go.py
"""
import subprocess
import sys

exporter = "../export_model.py"

models = [
    # regression
    ("model_lgb.txt",        "lgbm",      "GoModelTests/lgbm/model.go"),
    ("model_xgb.json",       "xgb",       "GoModelTests/xgb/model.go"),
    ("model_cb.cbm",         "cb",        "GoModelTests/cb/model.go"),
    # binary classification
    ("model_lgb_bin.txt",    "lgbmbin",   "GoModelTests/lgbmbin/model.go"),
    ("model_xgb_bin.json",   "xgbbin",    "GoModelTests/xgbbin/model.go"),
    ("model_cb_bin.cbm",     "cbbin",     "GoModelTests/cbbin/model.go"),
    # multiclass classification
    ("model_lgb_multi.txt",  "lgbmmulti", "GoModelTests/lgbmmulti/model.go"),
    ("model_xgb_multi.json", "xgbmulti",  "GoModelTests/xgbmulti/model.go"),
    ("model_cb_multi.cbm",   "cbmulti",   "GoModelTests/cbmulti/model.go"),
    # regression with missing values
    ("model_lgb_nan.txt",    "lgbmnan",   "GoModelTests/lgbmnan/model.go"),
    ("model_xgb_nan.json",   "xgbnan",    "GoModelTests/xgbnan/model.go"),
    ("model_cb_nan.cbm",     "cbnan",     "GoModelTests/cbnan/model.go"),
]

for model_path, pkg_name, output in models:
    print(f"Exporting {model_path} -> {output}")
    subprocess.run(
        [sys.executable, exporter, model_path, "-c", pkg_name, "-o", output,
         "--lang", "go"],
        check=True,
    )
    print()
