import pandas as pd

def assign_value_segment(row: pd.Series) -> str:
    if row.get("High_Value_Score", 0) >= 0.80:
        return "Very High Potential"

    if row.get("AddedWalletMoneyOnSignupDay", 0) == 1 or row.get("SignupIntentScore", 0) >= 5:
        return "High Intent"

    if row.get("PremiumDeviceFlag", 0) == 1 and row.get("ProfileCompleted", 0) == 1:
        return "Premium Profile"

    if row.get("ViewedPricingPage", 0) == 1 or row.get("ViewedServicePage", 0) == 1:
        return "Service Explorer"

    return "Low Early Signal"


def recommended_action(segment: str) -> str:
    actions = {
        "Very High Potential": "Prioritize for premium onboarding and high-touch CRM",
        "High Intent": "Send first-purchase offer or guided conversion journey",
        "Premium Profile": "Show premium plans, bundles, and personalized recommendations",
        "Service Explorer": "Send education content, trust signals, and product walkthrough",
        "Low Early Signal": "Run low-cost nurture and activation campaigns",
    }
    return actions.get(segment, "Run low-cost nurture and activation campaigns")
