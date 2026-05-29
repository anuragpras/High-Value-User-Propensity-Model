import os
import yaml
import joblib
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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


def decile_summary(scores, actual=None):
    df = pd.DataFrame({"Score": scores})
    df["Decile"] = 10 - pd.qcut(df["Score"].rank(method="first"), 10, labels=False)

    out = df.groupby("Decile").agg(
        Users=("Score", "count"),
        Min_Score=("Score", "min"),
        Max_Score=("Score", "max"),
        Avg_Score=("Score", "mean")
    ).reset_index()

    if actual is not None:
        df["Actual"] = actual.values
        hv = df.groupby("Decile")["Actual"].agg(["sum", "mean"]).reset_index()
        hv.columns = ["Decile", "High_Value_Users", "High_Value_Rate"]
        out = out.merge(hv, on="Decile", how="left")
        out["Capture_Rate"] = out["High_Value_Users"] / out["High_Value_Users"].sum()

    return out.round(4)


def main():
    cfg = load_config()
    target = cfg["project"]["target"]
    output_dir = cfg["paths"]["output_dir"]
    model_dir = cfg["paths"]["model_dir"]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_df = add_signup_value_features(pd.read_csv(cfg["paths"]["train_data"]))
    predict_df = add_signup_value_features(pd.read_csv(cfg["paths"]["predict_data"]))

    id_cols = ["UserID", "SignupDate"]
    leakage_cols = ["HighValueUser_90D", "Revenue_90D", "RechargeCount_90D"]
    features = [c for c in train_df.columns if c not in id_cols + leakage_cols]

    X = train_df[features]
    y = train_df[target]
    X_pred = predict_df[features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_state"],
        stratify=y
    )

    numeric_features = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features),
        ]
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=250,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=cfg["model"]["random_state"],
        n_jobs=-1,
        verbose=-1
    )

    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    train_scores = pipeline.predict_proba(X_train)[:, 1]
    test_scores = pipeline.predict_proba(X_test)[:, 1]
    pred_scores = pipeline.predict_proba(X_pred)[:, 1]

    threshold = cfg["model"]["threshold"]
    test_preds = (test_scores >= threshold).astype(int)

    metrics = pd.DataFrame([{
        "Target": target,
        "AUC_ROC": roc_auc_score(y_test, test_scores),
        "Accuracy": accuracy_score(y_test, test_preds),
        "Precision": precision_score(y_test, test_preds),
        "Recall": recall_score(y_test, test_preds),
        "F1_Score": f1_score(y_test, test_preds),
        "Threshold": threshold
    }]).round(4)

    out = predict_df[id_cols].copy()
    out["High_Value_Score"] = pred_scores.round(4)
    out["Score_Bucket"] = out["High_Value_Score"].apply(score_bucket)

    predict_df["High_Value_Score"] = out["High_Value_Score"]
    predict_df["Score_Bucket"] = out["Score_Bucket"]
    out["User_Value_Segment"] = predict_df.apply(assign_value_segment, axis=1)
    out["Recommended_Action"] = out["User_Value_Segment"].apply(recommended_action)
    out = out.sort_values("High_Value_Score", ascending=False)

    top_percent = cfg["outputs"]["top_percent"]
    top_n = max(1, int(len(out) * top_percent / 100))
    out.to_csv(os.path.join(output_dir, "all_scored_signup_users.csv"), index=False)
    out.head(top_n).to_csv(os.path.join(output_dir, f"top_{top_percent}pct_high_value_users.csv"), index=False)

    feature_importance = pd.DataFrame({
        "Feature": pipeline.named_steps["preprocess"].get_feature_names_out(),
        "Importance": pipeline.named_steps["model"].feature_importances_
    })
    feature_importance["Importance_Percent"] = feature_importance["Importance"] / feature_importance["Importance"].sum() * 100
    feature_importance = feature_importance.sort_values("Importance_Percent", ascending=False).head(30)

    feature_bucket = predict_df.groupby("Score_Bucket").agg(
        Users=("UserID", "count"),
        Avg_SignupIntentScore=("SignupIntentScore", "mean"),
        Avg_FirstSessionDurationSec=("FirstSessionDurationSec", "mean"),
        Avg_FirstSessionEventCount=("FirstSessionEventCount", "mean"),
        Avg_SignupWalletBalance=("SignupWalletBalance", "mean"),
        Pct_ProfileCompleted=("ProfileCompleted", "mean"),
        Pct_ViewedServicePage=("ViewedServicePage", "mean"),
        Pct_ViewedPricingPage=("ViewedPricingPage", "mean"),
        Pct_AddedWalletMoney=("AddedWalletMoneyOnSignupDay", "mean")
    ).reset_index().round(4)

    with pd.ExcelWriter(os.path.join(output_dir, "high_value_model_analysis_report.xlsx"), engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="Model_Metrics", index=False)
        feature_importance.to_excel(writer, sheet_name="Feature_Importance", index=False)
        decile_summary(train_scores, y_train).to_excel(writer, sheet_name="Train_Deciles", index=False)
        decile_summary(test_scores, y_test).to_excel(writer, sheet_name="Test_Deciles", index=False)
        decile_summary(pred_scores).to_excel(writer, sheet_name="Predict_Deciles", index=False)
        out.groupby(["Score_Bucket", "User_Value_Segment"]).size().reset_index(name="Users").to_excel(writer, sheet_name="Score_Bucket_Summary", index=False)
        feature_bucket.to_excel(writer, sheet_name="Feature_Bucket_Analysis", index=False)

    joblib.dump(pipeline, os.path.join(model_dir, "high_value_user_model.pkl"))
    print("High-value user propensity pipeline completed.")


if __name__ == "__main__":
    main()
