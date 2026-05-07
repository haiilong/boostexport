"""
Regenerate all three Go model files into GoModelTests/.
Run from the demo/ directory:
    python generate_go.py
"""
import subprocess
import sys

exporter = "../export_model.py"

models = [
    ("model_lgb.txt",  "lgbm", "GoModelTests/lgbm/model.go"),
    ("model_xgb.json", "xgb",  "GoModelTests/xgb/model.go"),
    ("model_cb.cbm",   "cb",   "GoModelTests/cb/model.go"),
]

for model_path, pkg_name, output in models:
    print(f"Exporting {model_path} -> {output}")
    subprocess.run(
        [sys.executable, exporter, model_path, "-c", pkg_name, "-o", output, "--lang", "go"],
        check=True,
    )
    print()
