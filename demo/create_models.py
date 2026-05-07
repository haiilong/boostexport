import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# --- Dataset ---
X, y = make_regression(
    n_samples=200, n_features=10, n_informative=6,
    noise=0.1, random_state=42
)
feature_cols = [f"f{i}" for i in range(10)]
df = pd.DataFrame(X, columns=feature_cols)
df["label"] = y
df.to_csv("data.csv", index=False)
print("Saved data.csv")

X_arr = df[feature_cols].values
y_arr = df["label"].values

# --- XGBoost ---
xgb_model = xgb.XGBRegressor(
    n_estimators=50, max_depth=3, learning_rate=0.1,
    objective="reg:squarederror", random_state=42
)
xgb_model.fit(X_arr, y_arr)
xgb_model.save_model("model_xgb.json")
pred_xgb = xgb_model.predict(X_arr)
print("XGBoost saved -> model_xgb.json")

# --- LightGBM ---
lgb_model = lgb.LGBMRegressor(
    n_estimators=50, max_depth=3, learning_rate=0.1,
    random_state=42, verbose=-1
)
lgb_model.fit(X_arr, y_arr)
lgb_model.booster_.save_model("model_lgb.txt")
pred_lgb = lgb_model.predict(X_arr)
print("LightGBM saved -> model_lgb.txt")

# --- CatBoost ---
cb_model = cb.CatBoostRegressor(
    iterations=50, depth=3, learning_rate=0.1,
    random_seed=42, verbose=0
)
cb_model.fit(X_arr, y_arr)
cb_model.save_model("model_cb.cbm")
pred_cb = cb_model.predict(X_arr)
print("CatBoost saved -> model_cb.cbm")

# --- Save ---
preds_df = pd.DataFrame({
    "label": y_arr,
    "pred_xgb": pred_xgb,
    "pred_lgb": pred_lgb,
    "pred_cb": pred_cb,
})
preds_df.to_csv("predictions_train.csv", index=False)
print("Saved predictions_train.csv")

print("\nRMSE summary:")
for name, preds in [("XGBoost", pred_xgb), ("LightGBM", pred_lgb), ("CatBoost", pred_cb)]:
    rmse = np.sqrt(np.mean((preds - y_arr) ** 2))
    print(f"  {name}: {rmse:.4f}")
