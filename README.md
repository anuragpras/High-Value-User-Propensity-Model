# High-Value User Propensity Model

## Overview

This repository contains an open-source machine learning pipeline for predicting whether a newly signed-up user is likely to become a **high-value user** in the future.

The model is designed to score users at or near the time of signup using early signals such as:

- Signup profile
- Acquisition source
- Device and app information
- First-session behavior
- Early intent actions
- Wallet and pricing signals

The output helps growth, CRM, product, and marketing teams identify high-potential users early and personalize onboarding or conversion journeys.

---

## Objective

Build a binary classification model to predict whether a user will become high value within a future window such as 30, 60, or 90 days.

In this sample repo, the target is:

```text
HighValueUser_90D

1 = User became high value within 90 days
0 = User did not become high value within 90 days
```

A high-value user can be defined using business logic such as:

- Revenue above a threshold
- Recharge amount above a threshold
- Multiple purchases or recharges
- High consultation usage
- High lifetime value

---

## Why Predict High Value at Signup?

Not all new users have the same long-term potential.

A signup-time propensity model helps teams:

- Prioritize high-potential users
- Improve onboarding personalization
- Allocate CRM budgets efficiently
- Identify premium users early
- Improve activation and long-term value
- Build better audience segments for paid marketing

---

## Feature Categories

### User Profile

- Age
- Gender
- City
- State
- Country
- Language

### Signup Context

- Signup Date
- Signup Day
- Signup Time

### Device and App

- Device Type
- Device Model
- Device Price Tier
- OS Details
- App Version
- ISP

### Acquisition

- Acquisition Source
- Campaign
- Adgroup
- First Event Name
- Acquisition Cost Bucket

### First-Session Behavior

- First Session Duration
- First Session Event Count
- Landing Page Viewed
- Profile Completed
- Notification Permission Given
- Viewed Service Page
- Viewed Pricing Page
- Used Search

### Early Wallet and Intent

- Signup Wallet Balance
- Added Wallet Money on Signup Day
- Signup Intent Score
- High Intent First Session Flag
- Premium Device Flag
- Paid Acquisition Flag

---

## Feature Engineering

Raw signup data may not directly capture user quality. This project creates derived features to identify early high-value signals.

### Signup Intent Score

```text
SignupIntentScore =
ProfileCompleted * 2
+ ViewedServicePage * 1.5
+ ViewedPricingPage * 1.2
+ UsedSearch * 1
+ AddedWalletMoneyOnSignupDay * 3
+ Long First Session Flag * 1.3
+ High Event Count Flag * 1
```

Higher values indicate stronger early user intent.

### Premium Device Flag

```text
PremiumDeviceFlag = 1 if DevicePriceTier is High or Premium
```

### High Intent First Session Flag

```text
HighIntentFirstSessionFlag = 1 if SignupIntentScore >= 4
```

### Paid Acquisition Flag

```text
PaidAcquisitionFlag = 1 if source is Google, Meta, or Affiliate
```

---

## User Value Segments

The model assigns users to business-readable segments.

| Segment | Meaning | Recommended Action |
|---|---|---|
| Very High Potential | Highest predicted high-value probability | Premium onboarding and high-touch CRM |
| High Intent | Strong early product or wallet intent | First-purchase offer or guided conversion journey |
| Premium Profile | Premium device/profile completion signals | Premium plans, bundles, and personalized recommendations |
| Service Explorer | Exploring service/pricing pages | Education content, trust signals, and walkthrough |
| Low Early Signal | Weak early activation behavior | Low-cost nurture and activation campaigns |

---

## Project Structure

```text
high-value-user-propensity-model/
│
├── data/
│   ├── sample_train.csv
│   └── sample_predict.csv
│
├── models/
│   └── MODEL_CARD.md
│
├── notebooks/
│   └── README.md
│
├── outputs/
│   ├── all_scored_signup_users.csv
│   ├── top_10pct_high_value_users.csv
│   └── high_value_model_analysis_report.xlsx
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── feature_engineering.py
│   ├── segmentation.py
│   └── generate_sample_data.py
│
├── config.yaml
├── requirements.txt
└── README.md
```

---

## Input Data

### Training Data

The training file should contain signup-time features and the target column:

```text
HighValueUser_90D
```

Optional future outcome columns can also be included for analysis:

```text
Revenue_90D
RechargeCount_90D
```

These are excluded from model training to avoid leakage.

### Prediction Data

The prediction file should contain signup-time features only.

Required ID columns:

```text
UserID
SignupDate
```

---

## Installation

```bash
git clone https://github.com/yourusername/high-value-user-propensity-model.git
cd high-value-user-propensity-model
pip install -r requirements.txt
```

---

## How To Run

### Step 1: Use sample data or replace it

Sample files are included:

```text
data/sample_train.csv
data/sample_predict.csv
```

### Step 2: Train and score

```bash
python src/train.py
```

Generated outputs:

```text
models/high_value_user_model.pkl
outputs/all_scored_signup_users.csv
outputs/top_10pct_high_value_users.csv
outputs/high_value_model_analysis_report.xlsx
```

### Step 3: Score a new signup batch

After training:

```bash
python src/predict.py
```

Generated output:

```text
outputs/new_batch_high_value_scores.csv
```

---

## Output Files

### all_scored_signup_users.csv

Contains:

- UserID
- SignupDate
- High_Value_Score
- Score_Bucket
- User_Value_Segment
- Recommended_Action

### top_10pct_high_value_users.csv

Top users ranked by predicted high-value probability.

### high_value_model_analysis_report.xlsx

Contains:

- Model_Metrics
- Threshold_Analysis
- Feature_Importance
- Train_Deciles
- Test_Deciles
- Predict_Deciles
- Score_Bucket_Summary
- Feature_Bucket_Analysis

---

## Example Use Cases

### Premium Onboarding

Give high-potential users a better onboarding journey.

### CRM Prioritization

Focus high-touch campaigns on users most likely to become valuable.

### Paid Marketing Optimization

Analyze which acquisition sources create high-value users.

### Personalization

Show different journeys to premium users, explorers, and low-signal users.

### Growth Analytics

Understand which signup-time behaviors predict future value.

---

## Production Rollout

Suggested pipeline:

```text
Signup Event / User Table
   ↓
Feature Engineering
   ↓
Python Model Inference
   ↓
User Value Score
   ↓
CRM / Onboarding / Product Personalization
```

Example rule:

```text
If High_Value_Score > 0.80 and Segment = Very High Potential,
trigger premium onboarding journey.
```

---

## Future Improvements

- SHAP explainability
- Revenue regression model
- Multi-window value prediction
- Uplift modeling
- Model monitoring
- Automated retraining
- Real-time API scoring
- Dashboard integration

---

## Disclaimer

The sample data is synthetic and does not contain real user information.

---

## License

MIT License
