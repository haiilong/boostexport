"""Train the full demo/test matrix: regression, binary, multiclass, and
NaN-regression models for XGBoost, LightGBM, and CatBoost.

Outputs (per task):
    data*.csv               feature matrix + label
    model_<lib>*.{json,txt,cbm}
    predictions*.csv        expected outputs from the Python models
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

FEATURES = 10
feature_cols = [f"f{i}" for i in range(FEATURES)]


def save_data(path: str, X: np.ndarray, y: np.ndarray) -> None:
    df = pd.DataFrame(X, columns=feature_cols)
    df["label"] = y
    df.to_csv(path, index=False, na_rep="NaN")
    print(f"Saved {path}")


# =========================================================================
# 1. Regression (the original demo dataset)
# =========================================================================
X, y = make_regression(
    n_samples=200, n_features=FEATURES, n_informative=6,
    noise=0.1, random_state=42
)
save_data("data.csv", X, y)

xgb_model = xgb.XGBRegressor(
    n_estimators=50, max_depth=3, learning_rate=0.1,
    objective="reg:squarederror", random_state=42
)
xgb_model.fit(X, y)
xgb_model.save_model("model_xgb.json")

lgb_model = lgb.LGBMRegressor(
    n_estimators=50, max_depth=3, learning_rate=0.1,
    random_state=42, verbose=-1
)
lgb_model.fit(X, y)
lgb_model.booster_.save_model("model_lgb.txt")

cb_model = cb.CatBoostRegressor(
    iterations=50, depth=3, learning_rate=0.1,
    random_seed=42, verbose=0
)
cb_model.fit(X, y)
cb_model.save_model("model_cb.cbm")

pd.DataFrame({
    "label": y,
    "pred_xgb": xgb_model.predict(X),
    "pred_lgb": lgb_model.predict(X),
    "pred_cb": cb_model.predict(X),
}).to_csv("predictions_train.csv", index=False)
print("Saved predictions_train.csv (regression)")

# =========================================================================
# 2. Binary classification (imbalanced, so XGBoost's auto base_score != 0.5)
# =========================================================================
Xb, yb = make_classification(
    n_samples=300, n_features=FEATURES, n_informative=6,
    weights=[0.65], flip_y=0.05, class_sep=0.8, random_state=42
)
save_data("data_binary.csv", Xb, yb)

xgb_bin = xgb.XGBClassifier(
    n_estimators=40, max_depth=3, learning_rate=0.1, random_state=42
)
xgb_bin.fit(Xb, yb)
xgb_bin.save_model("model_xgb_bin.json")

lgb_bin = lgb.LGBMClassifier(
    n_estimators=40, max_depth=3, learning_rate=0.1,
    random_state=42, verbose=-1
)
lgb_bin.fit(Xb, yb)
lgb_bin.booster_.save_model("model_lgb_bin.txt")

cb_bin = cb.CatBoostClassifier(
    iterations=40, depth=3, learning_rate=0.1, random_seed=42, verbose=0
)
cb_bin.fit(Xb, yb)
cb_bin.save_model("model_cb_bin.cbm")

pd.DataFrame({
    "label": yb,
    "score_xgb": xgb_bin.get_booster().predict(
        xgb.DMatrix(Xb), output_margin=True
    ).astype(float),
    "proba_xgb": xgb_bin.predict_proba(Xb)[:, 1],
    "score_lgb": lgb_bin.booster_.predict(Xb, raw_score=True),
    "proba_lgb": lgb_bin.predict_proba(Xb)[:, 1],
    "score_cb": cb_bin.predict(Xb, prediction_type="RawFormulaVal"),
    "proba_cb": cb_bin.predict_proba(Xb)[:, 1],
}).to_csv("predictions_binary.csv", index=False)
print("Saved predictions_binary.csv (binary)")

# =========================================================================
# 3. Multiclass classification (3 classes)
# =========================================================================
Xm, ym = make_classification(
    n_samples=300, n_features=FEATURES, n_informative=6, n_classes=3,
    n_clusters_per_class=1, class_sep=1.0, random_state=42
)
save_data("data_multi.csv", Xm, ym)

xgb_multi = xgb.XGBClassifier(
    n_estimators=30, max_depth=3, learning_rate=0.1, random_state=42
)
xgb_multi.fit(Xm, ym)
xgb_multi.save_model("model_xgb_multi.json")

lgb_multi = lgb.LGBMClassifier(
    n_estimators=30, max_depth=3, learning_rate=0.1,
    random_state=42, verbose=-1
)
lgb_multi.fit(Xm, ym)
lgb_multi.booster_.save_model("model_lgb_multi.txt")

cb_multi = cb.CatBoostClassifier(
    iterations=30, depth=3, learning_rate=0.1, random_seed=42, verbose=0,
    loss_function="MultiClass"
)
cb_multi.fit(Xm, ym)
cb_multi.save_model("model_cb_multi.cbm")

multi_out = pd.DataFrame({"label": ym})
for lib, proba in [
    ("xgb", xgb_multi.predict_proba(Xm)),
    ("lgb", lgb_multi.predict_proba(Xm)),
    ("cb", cb_multi.predict_proba(Xm)),
]:
    for c in range(proba.shape[1]):
        multi_out[f"proba_{lib}_{c}"] = proba[:, c]
multi_out.to_csv("predictions_multi.csv", index=False)
print("Saved predictions_multi.csv (multiclass)")

# =========================================================================
# 4. Regression with missing values (NaN routing)
# =========================================================================
rng = np.random.default_rng(42)
Xn, yn = make_regression(
    n_samples=300, n_features=FEATURES, n_informative=6,
    noise=0.1, random_state=7
)
Xn[rng.random(Xn.shape) < 0.25] = np.nan
# a fully-missing row and a fully-zero row (exercises Zero/None missing types)
Xn[0, :] = np.nan
Xn[1, :] = 0.0
save_data("data_nan.csv", Xn, yn)

xgb_nan = xgb.XGBRegressor(
    n_estimators=40, max_depth=3, learning_rate=0.1, random_state=42
)
xgb_nan.fit(Xn, yn)
xgb_nan.save_model("model_xgb_nan.json")

lgb_nan = lgb.LGBMRegressor(
    n_estimators=40, max_depth=3, learning_rate=0.1,
    random_state=42, verbose=-1
)
lgb_nan.fit(Xn, yn)
lgb_nan.booster_.save_model("model_lgb_nan.txt")

# nan_mode="Max" exercises the "NaN goes to the true side" treatment
cb_nan = cb.CatBoostRegressor(
    iterations=40, depth=3, learning_rate=0.1, random_seed=42, verbose=0,
    nan_mode="Max"
)
cb_nan.fit(Xn, yn)
cb_nan.save_model("model_cb_nan.cbm")

pd.DataFrame({
    "label": yn,
    "pred_xgb": xgb_nan.predict(Xn).astype(float),
    "pred_lgb": lgb_nan.predict(Xn),
    "pred_cb": cb_nan.predict(Xn),
}).to_csv("predictions_nan.csv", index=False)
print("Saved predictions_nan.csv (NaN regression)")

print("\nDone: 12 models, 4 datasets, 4 prediction files.")
