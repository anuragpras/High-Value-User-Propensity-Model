import pandas as pd

def add_signup_value_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SignupIntentScore"] = (
        df["ProfileCompleted"] * 2
        + df["ViewedServicePage"] * 1.5
        + df["ViewedPricingPage"] * 1.2
        + df["UsedSearch"] * 1.0
        + df["AddedWalletMoneyOnSignupDay"] * 3
        + (df["FirstSessionDurationSec"] > 240).astype(int) * 1.3
        + (df["FirstSessionEventCount"] > 10).astype(int) * 1.0
    ).round(2)

    df["PremiumDeviceFlag"] = df["DevicePriceTier"].isin(["High", "Premium"]).astype(int)
    df["HighIntentFirstSessionFlag"] = (df["SignupIntentScore"] >= 4).astype(int)
    df["PaidAcquisitionFlag"] = df["AcquisitionSource"].isin(["google", "meta", "affiliate"]).astype(int)

    return df
