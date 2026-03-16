"""
predict_churn.py — Production Inference Script
===============================================
Usage:
    python predict_churn.py                        # runs demo with 3 sample clients
    python predict_churn.py --balance 1200 --age 32 --days_tx 28 --balance_change -800

The script loads the trained Gradient Boosting model and predicts churn probability
for one or more clients. It also outputs the risk segment and recommended action.

Requirements:
    pip install joblib pandas scikit-learn
"""

import joblib
import pandas as pd
import numpy as np
import argparse
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "gb_model.pkl")
META_PATH     = os.path.join(os.path.dirname(__file__), "model_meta.pkl")

# Optimal profit-based threshold (computed in training pipeline)
# Default 0.5 maximises accuracy; 0.11 maximises expected profit.
THRESHOLD = 0.11

# Risk tiers
RISK_TIERS = [
    (0.75, "🚨 CRITICAL", "Urgent: personal manager call + fee waiver + premium upgrade"),
    (0.50, "🔴 HIGH",     "Proactive outreach call — discuss pain points, make tailored offer"),
    (0.25, "🟡 MEDIUM",   "Soft touch — personalised email / product recommendation"),
    (0.00, "🟢 LOW",      "Standard service — no intervention needed"),
]

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run churn_analysis.py first to train and save the model."
        )
    model = joblib.load(MODEL_PATH)
    meta  = joblib.load(META_PATH) if os.path.exists(META_PATH) else {}
    return model, meta


# ── FEATURE ENGINEERING (must match training pipeline exactly) ────────────────
def engineer_features(client_dict: dict) -> pd.DataFrame:
    """
    Takes a dict of raw client fields and returns a DataFrame
    with all 24 engineered features in the correct column order.

    Minimum required fields:
        age, gender_enc, occupation_enc, vintage, dependents, city,
        customer_nw_category, current_balance, previous_month_end_balance,
        average_monthly_balance_prevQ, average_monthly_balance_prevQ2,
        current_month_credit, previous_month_credit,
        current_month_debit, previous_month_debit,
        current_month_balance, previous_month_balance, days_since_last_tx
    """
    d = client_dict.copy()

    # Engineered features
    d['balance_change']        = d.get('current_balance', 0) - d.get('previous_month_end_balance', 0)
    d['credit_debit_ratio']    = min(d.get('current_month_credit', 0) /
                                     (d.get('current_month_debit', 0) + 1), 20)
    d['avg_balance_trend']     = (d.get('average_monthly_balance_prevQ', 0) -
                                   d.get('average_monthly_balance_prevQ2', 0))
    d['low_balance_flag']      = int(d.get('current_balance', 0) < 500)
    d['debit_activity_ratio']  = min(d.get('current_month_debit', 0) /
                                     (d.get('previous_month_debit', 0) + 1), 10)
    d['credit_activity_trend'] = min(d.get('current_month_credit', 0) /
                                     (d.get('previous_month_credit', 0) + 1), 10)

    # Feature order must match training data exactly
    feature_order = [
        'age', 'vintage', 'dependents', 'city', 'customer_nw_category',
        'current_balance', 'previous_month_end_balance',
        'average_monthly_balance_prevQ', 'average_monthly_balance_prevQ2',
        'current_month_credit', 'previous_month_credit',
        'current_month_debit', 'previous_month_debit',
        'current_month_balance', 'previous_month_balance',
        'days_since_last_tx', 'gender_enc', 'occupation_enc',
        'balance_change', 'credit_debit_ratio', 'avg_balance_trend',
        'low_balance_flag', 'debit_activity_ratio', 'credit_activity_trend',
    ]

    # Fill any missing features with 0 (safe default)
    for col in feature_order:
        if col not in d:
            d[col] = 0

    return pd.DataFrame([{col: d[col] for col in feature_order}])


# ── PREDICT ───────────────────────────────────────────────────────────────────
def predict_client(client_dict: dict, model, threshold: float = THRESHOLD) -> dict:
    """
    Predict churn probability for a single client.

    Args:
        client_dict: dict of client features (raw + will be engineered)
        model: trained sklearn model with predict_proba
        threshold: classification threshold (default = profit-optimal 0.11)

    Returns:
        dict with keys: prob, prediction, risk_tier, action
    """
    X = engineer_features(client_dict)
    prob = float(model.predict_proba(X)[0][1])
    prediction = int(prob >= threshold)

    # Determine risk tier
    risk_tier, action = "🟢 LOW", "Standard service"
    for cutoff, tier, act in RISK_TIERS:
        if prob >= cutoff:
            risk_tier, action = tier, act
            break

    return {
        "churn_probability": round(prob, 4),
        "churn_prediction":  "CHURN" if prediction else "STAY",
        "risk_tier":         risk_tier,
        "recommended_action": action,
        "threshold_used":    threshold,
    }


# ── BATCH PREDICTION ──────────────────────────────────────────────────────────
def predict_batch(clients: list, model, threshold: float = THRESHOLD) -> pd.DataFrame:
    """
    Predict churn for a list of client dicts.
    Returns a DataFrame sorted by churn_probability descending.
    """
    rows = []
    for i, client in enumerate(clients):
        result = predict_client(client, model, threshold)
        rows.append({"client_id": i + 1, **result})
    return pd.DataFrame(rows).sort_values("churn_probability", ascending=False)


# ── DEMO ──────────────────────────────────────────────────────────────────────
DEMO_CLIENTS = [
    {
        # HIGH RISK: young, low balance, actively withdrawing
        "name":                          "Client A — High Risk",
        "age":                           32,
        "vintage":                       1200,
        "current_balance":               400,
        "previous_month_end_balance":    3200,
        "average_monthly_balance_prevQ": 2500,
        "average_monthly_balance_prevQ2":3800,
        "current_month_credit":          500,
        "previous_month_credit":         2000,
        "current_month_debit":           2800,
        "previous_month_debit":          1500,
        "current_month_balance":         400,
        "previous_month_balance":        3200,
        "days_since_last_tx":            28,
        "gender_enc":                    0,
        "occupation_enc":                2,   # self_employed
        "customer_nw_category":          1,
        "dependents":                    0,
        "city":                          1,
    },
    {
        # MEDIUM RISK: mid-tenure, stable but declining
        "name":                          "Client B — Medium Risk",
        "age":                           45,
        "vintage":                       1800,
        "current_balance":               8500,
        "previous_month_end_balance":    9200,
        "average_monthly_balance_prevQ": 9000,
        "average_monthly_balance_prevQ2":9800,
        "current_month_credit":          1800,
        "previous_month_credit":         2200,
        "current_month_debit":           2100,
        "previous_month_debit":          2000,
        "current_month_balance":         8500,
        "previous_month_balance":        9200,
        "days_since_last_tx":            45,
        "gender_enc":                    1,
        "occupation_enc":                3,   # salaried
        "customer_nw_category":          2,
        "dependents":                    2,
        "city":                          3,
    },
    {
        # LOW RISK: loyal, high balance, active deposits
        "name":                          "Client C — Low Risk",
        "age":                           55,
        "vintage":                       2800,
        "current_balance":               42000,
        "previous_month_end_balance":    40000,
        "average_monthly_balance_prevQ": 39000,
        "average_monthly_balance_prevQ2":37500,
        "current_month_credit":          12000,
        "previous_month_credit":         10000,
        "current_month_debit":           8000,
        "previous_month_debit":          8500,
        "current_month_balance":         42000,
        "previous_month_balance":        40000,
        "days_since_last_tx":            5,
        "gender_enc":                    0,
        "occupation_enc":                4,   # retired
        "customer_nw_category":          3,
        "dependents":                    3,
        "city":                          2,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Predict customer churn probability")
    parser.add_argument("--balance",        type=float, default=None)
    parser.add_argument("--age",            type=int,   default=None)
    parser.add_argument("--days_tx",        type=int,   default=None, dest="days_since_last_tx")
    parser.add_argument("--balance_change", type=float, default=None)
    parser.add_argument("--threshold",      type=float, default=THRESHOLD)
    args = parser.parse_args()

    print("=" * 62)
    print("  BANK CHURN PREDICTION — Production Inference")
    print(f"  Model: Gradient Boosting  |  Threshold: {args.threshold}")
    print("=" * 62)

    model, meta = load_model()
    print(f"[✓] Model loaded from {MODEL_PATH}\n")

    if args.balance is not None:
        # Single client from CLI arguments
        client = {
            "current_balance":            args.balance,
            "previous_month_end_balance": args.balance - (args.balance_change or 0),
            "age":                        args.age or 35,
            "days_since_last_tx":         args.days_since_last_tx or 30,
            "balance_change":             args.balance_change or 0,
        }
        result = predict_client(client, model, args.threshold)
        print(f"  Churn Probability : {result['churn_probability']:.2%}")
        print(f"  Prediction        : {result['churn_prediction']}")
        print(f"  Risk Tier         : {result['risk_tier']}")
        print(f"  Recommended Action: {result['recommended_action']}")
    else:
        # Run demo with 3 example clients
        print("  Running demo with 3 sample clients...\n")
        for client in DEMO_CLIENTS:
            name = client.pop("name", "Client")
            result = predict_client(client, model, args.threshold)
            print(f"  ── {name}")
            print(f"     Age: {client['age']}  Balance: {client['current_balance']:,}"
                  f"  Days inactive: {client['days_since_last_tx']}")
            print(f"     Churn Probability : {result['churn_probability']:.2%}")
            print(f"     Prediction        : {result['churn_prediction']}")
            print(f"     Risk Tier         : {result['risk_tier']}")
            print(f"     Action            : {result['recommended_action']}")
            print()

    print("=" * 62)
    print("  Usage: python predict_churn.py --balance 1200 --age 32")
    print("         --days_tx 28 --balance_change -800")
    print("=" * 62)


if __name__ == "__main__":
    main()
