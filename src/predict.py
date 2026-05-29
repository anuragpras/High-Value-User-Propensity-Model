import os
import yaml
import joblib
import pandas as pd

from feature_engineering import add_signup_value_features
from segmentation import assign_value_segment, recommended_action


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def score_bucket(score):
    if score >= 0.80:
        return "0.80 - 1.00"
    if score >= 0.60:
        return "0.60 - 0.80"
    if score >= 0.40:
        return "0.40 - 0.60"
    if score >= 0.20:
        return "0.20 - 0.40"
    return "0.00 - 0.20"


def main():
    cfg = load_config()
    model_path = os.path.join(cfg["paths"]["model_dir"], "high_value_user_model.pkl")
    model = joblib.load(model_path)

    df = add_signup_value_features(pd.read_csv(cfg["paths"]["predict_data"]))
    id_cols = ["UserID", "SignupDate"]
    features = [c for c in df.columns if c not in id_cols + ["HighValueUser_90D", "Revenue_90D", "RechargeCount_90D"]]

    scores = model.predict_proba(df[features])[:, 1]

    out = df[id_cols].copy()
    out["High_Value_Score"] = scores.round(4)
    out["Score_Bucket"] = out["High_Value_Score"].apply(score_bucket)

    df["High_Value_Score"] = out["High_Value_Score"]
    df["Score_Bucket"] = out["Score_Bucket"]
    out["User_Value_Segment"] = df.apply(assign_value_segment, axis=1)
    out["Recommended_Action"] = out["User_Value_Segment"].apply(recommended_action)

    out = out.sort_values("High_Value_Score", ascending=False)
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    out.to_csv(os.path.join(cfg["paths"]["output_dir"], "new_batch_high_value_scores.csv"), index=False)
    print("Prediction complete.")


if __name__ == "__main__":
    main()
