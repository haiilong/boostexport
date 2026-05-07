import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

TOLERANCE = 1e-4
feature_cols = [f"f{i}" for i in range(10)]

df = pd.read_csv("data.csv")
X_arr = df[feature_cols].values
y_arr = df["label"].values

preds_saved = pd.read_csv("predictions_train.csv")

def check(name, pred_new, pred_saved_col):
    diff = np.abs(pred_new - preds_saved[pred_saved_col].values)
    match = (diff < TOLERANCE).all()
    print(f"{name}: max_diff={diff.max():.2e}  match={match}")

# XGBoost
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("model_xgb.json")
pred_xgb = xgb_model.predict(X_arr)
check("XGBoost ", pred_xgb, "pred_xgb")

# LightGBM
lgb_booster = lgb.Booster(model_file="model_lgb.txt")
pred_lgb = lgb_booster.predict(X_arr)
check("LightGBM", pred_lgb, "pred_lgb")

# CatBoost
cb_model = cb.CatBoostRegressor()
cb_model.load_model("model_cb.cbm")
pred_cb = cb_model.predict(X_arr)
check("CatBoost", pred_cb, "pred_cb")

print("\nRMSE on loaded models:")
for name, preds in [("XGBoost", pred_xgb), ("LightGBM", pred_lgb), ("CatBoost", pred_cb)]:
    rmse = np.sqrt(np.mean((preds - y_arr) ** 2))
    print(f"  {name}: {rmse:.4f}")
