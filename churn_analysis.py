# =============================================================================
# BANK CUSTOMER CHURN ANALYSIS  —  PROFESSIONAL ML PIPELINE
# =============================================================================
# Dataset  : churn_prediction.csv  (28,382 clients, 21 raw features)
# Target   : churn  (1 = left, 0 = stayed)  |  Churn rate: 18.53%
# Split    : 70 / 15 / 15  (train / validation / test)
# CV       : 5-fold StratifiedKFold on training set
# Models   : Logistic Regression (Pipeline) · Random Forest · Gradient Boosting
# Tuning   : RandomizedSearchCV on Random Forest
# Explain  : Feature Importance (RF, GB) + Coefficients (LR) + Permutation Imp.
# Business : Risk segmentation · Retention simulation · Error cost analysis
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

from sklearn.model_selection  import (train_test_split, StratifiedKFold,
                                       cross_validate, RandomizedSearchCV)
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection       import permutation_importance
from sklearn.metrics          import (accuracy_score, precision_score, recall_score,
                                       f1_score, roc_auc_score, roc_curve,
                                       precision_recall_curve, confusion_matrix,
                                       average_precision_score)
from scipy.stats import randint, uniform

warnings.filterwarnings('ignore')

# ── PALETTE ───────────────────────────────────────────────────────────────────
TEAL  = '#0D9488'; ROSE  = '#E11D48'; IND   = '#6366F1'
AMBER = '#F59E0B'; SLATE = '#64748B'; DARK  = '#1E293B'; BG = '#F8FAFC'
GREEN = '#10B981'; PURPLE= '#8B5CF6'
PALETTE = [TEAL, ROSE, IND, AMBER, SLATE, GREEN, PURPLE]

plt.rcParams.update({
    'figure.facecolor': BG,   'axes.facecolor': BG,
    'axes.edgecolor':  '#CBD5E1', 'axes.titlecolor': DARK,
    'axes.labelcolor': SLATE,  'xtick.color': SLATE, 'ytick.color': SLATE,
    'font.family': 'DejaVu Sans', 'figure.dpi': 150,
})

OUT = '/home/claude'   # output directory for all saved images

print("=" * 70)
print("  BANK CUSTOMER CHURN — PROFESSIONAL ML PIPELINE")
print("=" * 70)


# =============================================================================
# 1. DATA LOADING
# =============================================================================

df = pd.read_csv('/mnt/user-data/uploads/churn_prediction.csv')
assert df.shape == (28382, 21), f"Unexpected shape: {df.shape}"
print(f"\n[✓] Loaded : {df.shape[0]:,} rows · {df.shape[1]} columns")
print(f"    Churn  : {df['churn'].mean():.2%}  ({df['churn'].sum():,} churned)")


# =============================================================================
# 2. DATA CLEANING
# =============================================================================
print("\n── 2. CLEANING ───────────────────────────────────────────────────────")

# Drop non-informative admin columns
df.drop(columns=['customer_id', 'branch_code'], inplace=True)

# Convert last_transaction date → days since last activity (recency feature)
# Reference = max date in dataset (acting as "today")
df['last_transaction']   = pd.to_datetime(df['last_transaction'], errors='coerce')
ref_date                 = df['last_transaction'].max()
df['days_since_last_tx'] = (ref_date - df['last_transaction']).dt.days
df.drop(columns=['last_transaction'], inplace=True)

# Impute missing values
# — Numeric columns: median (robust to outliers)
# — Categorical: mode (most frequent value)
# — days_since_last_tx NaN → max (client with NO transaction = maximally inactive)
for col in ['dependents', 'city']:
    df[col] = df[col].fillna(df[col].median())
for col in ['gender', 'occupation']:
    df[col] = df[col].fillna(df[col].mode()[0])
df['days_since_last_tx'] = df['days_since_last_tx'].fillna(df['days_since_last_tx'].max())

print(f"[✓] Cleaned  · NaN remaining: {df.isnull().sum().sum()}")


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
# Why these features?
#
#  balance_change       — Captures DIRECTION of financial movement.
#                         A large negative value means the client is actively
#                         withdrawing money — the clearest early warning of churn.
#
#  credit_debit_ratio   — Net cash flow: ratio > 1 means more coming in than going out.
#                         Ratio < 1 signals net outflow (client spending down the account).
#                         +1 prevents division-by-zero.  Clipped at 20 to kill outliers.
#
#  avg_balance_trend    — Quarter-over-quarter balance change.
#                         Negative = the client is slowly disengaging from the bank.
#
#  low_balance_flag     — Binary: balance < 500.  Small financial commitment = easy to leave.
#
#  debit_activity_ratio — Current month debit ÷ previous month debit.
#                         Rising ratio = increasing outflow activity (withdrawal trend).
#
#  credit_activity_trend— Current month credit ÷ previous month credit.
#                         Declining ratio = client depositing less → disengaging.

print("\n── 3. FEATURE ENGINEERING ────────────────────────────────────────────")

df['balance_change']         = df['current_balance'] - df['previous_month_end_balance']
df['credit_debit_ratio']     = (df['current_month_credit'] / (df['current_month_debit'] + 1)).clip(0, 20)
df['avg_balance_trend']      = df['average_monthly_balance_prevQ'] - df['average_monthly_balance_prevQ2']
df['low_balance_flag']       = (df['current_balance'] < 500).astype(int)
# NEW: debit activity ratio — rising debit = client withdrawing more
df['debit_activity_ratio']   = (df['current_month_debit']  / (df['previous_month_debit']  + 1)).clip(0, 10)
# NEW: credit activity trend — falling credit = client depositing less
df['credit_activity_trend']  = (df['current_month_credit'] / (df['previous_month_credit'] + 1)).clip(0, 10)

print(f"[✓] 6 engineered features created")

# Encode categorical columns
le = LabelEncoder()
df['gender_enc']     = le.fit_transform(df['gender'].astype(str))
df['occupation_enc'] = le.fit_transform(df['occupation'].astype(str))
df_clean = df.drop(columns=['gender', 'occupation'])   # keep clean copy for EDA

# Final NaN sweep (arithmetic on NaN columns can propagate)
for col in df_clean.columns:
    if df_clean[col].isnull().any():
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

print(f"[✓] Encoding done · NaN after sweep: {df_clean.isnull().sum().sum()}")


# =============================================================================
# 4. TRAIN / VALIDATION / TEST SPLIT  (70 / 15 / 15)
# =============================================================================
# Why three-way split?
#   • Train      — model learns patterns
#   • Validation — tune hyperparameters without touching test data
#   • Test       — final unbiased evaluation, reported ONCE
# Using only train/test risks overfitting to test during tuning.

print("\n── 4. TRAIN / VAL / TEST SPLIT (70 / 15 / 15) ───────────────────────")

X = df_clean.drop(columns=['churn'])
y = df_clean['churn']

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
# Second split: 50/50 on the 30% → 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"[✓] Train      : {len(X_train):,}  ({len(X_train)/len(X):.0%})  churn: {y_train.mean():.2%}")
print(f"[✓] Validation : {len(X_val):,}   ({len(X_val)/len(X):.0%})  churn: {y_val.mean():.2%}")
print(f"[✓] Test       : {len(X_test):,}   ({len(X_test)/len(X):.0%})  churn: {y_test.mean():.2%}")
print(f"[✓] Features   : {X.shape[1]}")


# =============================================================================
# 5. CROSS-VALIDATION (5-fold, on TRAIN set only)
# =============================================================================
# 5-fold stratified CV gives a more robust estimate of model performance than
# a single train/val split. We run CV on the training set, then evaluate
# final models on the held-out test set.

print("\n── 5. 5-FOLD CROSS-VALIDATION ────────────────────────────────────────")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression inside a Pipeline: scaler + model in one object.
# Pipeline prevents data leakage — scaler is fit only on each train fold.
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr',     LogisticRegression(max_iter=1000, random_state=42,
                                  class_weight='balanced'))
])

rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=12, random_state=42,
    class_weight='balanced', n_jobs=-1
)

gb_model = GradientBoostingClassifier(
    n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
)

cv_models = {
    'Logistic Regression': lr_pipe,
    'Random Forest':       rf_model,
    'Gradient Boosting':   gb_model,
}

cv_results = {}
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

for name, model in cv_models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv,
                            scoring=scoring, n_jobs=-1)
    cv_results[name] = {k: scores[f'test_{k}'] for k in scoring}
    auc_mean = cv_results[name]['roc_auc'].mean()
    f1_mean  = cv_results[name]['f1'].mean()
    print(f"  {name:<25}  AUC={auc_mean:.3f}±{cv_results[name]['roc_auc'].std():.3f}"
          f"  F1={f1_mean:.3f}±{cv_results[name]['f1'].std():.3f}")


# =============================================================================
# 6. HYPERPARAMETER TUNING  (RandomizedSearchCV on Random Forest)
# =============================================================================
# We tune RF because it has the highest CV AUC and is the deployment candidate.
# RandomizedSearch is faster than GridSearch for large param spaces.
# Search is run on train set; best params are validated on val set.

print("\n── 6. HYPERPARAMETER TUNING (RF, RandomizedSearchCV) ────────────────")

param_dist = {
    'n_estimators': randint(100, 400),    # number of trees
    'max_depth':    [8, 10, 12, 15, None],# tree depth
    'min_samples_split': randint(2, 20),  # min samples to split a node
    'min_samples_leaf':  randint(1, 10),  # min samples in a leaf
    'max_features': ['sqrt', 'log2', 0.5],# features per split
}

rscv = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
    param_distributions=param_dist,
    n_iter=30,           # evaluate 30 random configurations
    cv=cv,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)
rscv.fit(X_train, y_train)

print(f"[✓] Best params  : {rscv.best_params_}")
print(f"[✓] Best CV AUC  : {rscv.best_score_:.4f}")

# Use tuned RF as the primary model going forward
rf_tuned = rscv.best_estimator_

# Validate on validation set
rf_val_auc = roc_auc_score(y_val, rf_tuned.predict_proba(X_val)[:, 1])
print(f"[✓] Validation AUC (tuned RF): {rf_val_auc:.4f}")


# =============================================================================
# 7. FINAL MODEL TRAINING & TEST EVALUATION
# =============================================================================
print("\n── 7. FINAL MODEL EVALUATION (Test Set) ─────────────────────────────")

# Retrain all three models on full train set before test evaluation
lr_pipe.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
# rf_tuned was already trained by RandomizedSearchCV

final_models = {
    'Logistic Regression': lr_pipe,
    'Random Forest (tuned)': rf_tuned,
    'Gradient Boosting':   gb_model,
}

test_results = {}
for name, model in final_models.items():
    yp   = model.predict(X_test)
    yprob= model.predict_proba(X_test)[:, 1]
    test_results[name] = {
        'model': model, 'y_pred': yp, 'y_prob': yprob,
        'Accuracy' : accuracy_score(y_test, yp),
        'Precision': precision_score(y_test, yp),
        'Recall'   : recall_score(y_test, yp),
        'F1'       : f1_score(y_test, yp),
        'ROCAUC'   : roc_auc_score(y_test, yprob),
        'AvgPrec'  : average_precision_score(y_test, yprob),
    }
    r = test_results[name]
    print(f"\n  {name}")
    print(f"    Accuracy={r['Accuracy']:.4f}  Precision={r['Precision']:.4f}"
          f"  Recall={r['Recall']:.4f}  F1={r['F1']:.4f}"
          f"  AUC={r['ROCAUC']:.4f}  AvgPrec={r['AvgPrec']:.4f}")

# Best model for business analysis = highest AUC
BEST = max(test_results, key=lambda k: test_results[k]['ROCAUC'])
print(f"\n[★] Best model: {BEST}")


# =============================================================================
# 8. EDA  —  DEEP DIVE INTO KEY FEATURES
# =============================================================================
print("\n── 8. EDA CHARTS ─────────────────────────────────────────────────────")

# ── 8a. Overview: 6-panel ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Bank Churn EDA — Real Data  |  28,382 clients  |  18.53% churn',
             fontsize=14, fontweight='bold', color=DARK, y=1.01)

# Churn distribution
ax = axes[0, 0]
counts = df['churn'].value_counts().sort_index()
bars = ax.bar(['Stayed', 'Churned'], counts.values,
              color=[TEAL, ROSE], edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+60,
            f'{val:,}\n({val/len(df):.1%})', ha='center', va='bottom',
            fontsize=10, color=DARK, fontweight='bold')
ax.set_title('Churn Distribution', fontweight='bold')
ax.set_ylabel('Clients'); ax.set_ylim(0, counts.max()*1.2)

# Age distribution
ax = axes[0, 1]
for cv_, lbl in [(0,'Stayed'),(1,'Churned')]:
    ax.hist(df[df['churn']==cv_]['age'], bins=35, alpha=0.65, label=lbl,
            color=[TEAL,ROSE][cv_], edgecolor='white', linewidth=0.4)
ax.set_title('Age Distribution', fontweight='bold')
ax.set_xlabel('Age'); ax.legend(fontsize=9)

# Churn by occupation
ax = axes[0, 2]
occ = df.groupby('occupation')['churn'].mean().sort_values(ascending=True)
ax.barh(occ.index, occ.values*100,
        color=[PALETTE[i%len(PALETTE)] for i in range(len(occ))],
        edgecolor='white', linewidth=1)
for bar, val in zip(ax.patches, occ.values):
    ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
            f'{val:.1%}', va='center', fontsize=9)
ax.set_title('Churn Rate by Occupation', fontweight='bold')
ax.set_xlabel('Churn Rate (%)')

# Churn by NW category
ax = axes[1, 0]
nw = df.groupby('customer_nw_category')['churn'].mean()*100
ax.bar([f'NW {int(x)}' for x in nw.index], nw.values,
       color=[TEAL, AMBER, ROSE][:len(nw)], edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(ax.patches, nw.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Churn by NW Category', fontweight='bold')
ax.set_ylabel('Churn Rate (%)'); ax.set_ylim(0, nw.max()*1.3)

# Days since last tx
ax = axes[1, 1]
for cv_, lbl in [(0,'Stayed'),(1,'Churned')]:
    ax.hist(df[df['churn']==cv_]['days_since_last_tx'].clip(0,500), bins=30,
            alpha=0.65, label=lbl, color=[TEAL,ROSE][cv_], edgecolor='white', linewidth=0.4)
ax.set_title('Days Since Last Transaction', fontweight='bold')
ax.set_xlabel('Days'); ax.legend(fontsize=9)

# Vintage
ax = axes[1, 2]
bins_v=[0,1000,1500,2000,2200,2400,3000]
lbls_v=['<1K','1K-1.5K','1.5K-2K','2K-2.2K','2.2K-2.4K','>2.4K']
df['vbin'] = pd.cut(df['vintage'], bins=bins_v, labels=lbls_v)
vc = df.groupby('vbin', observed=False)['churn'].mean()*100
ax.plot(range(len(vc)), vc.values, marker='o', color=IND, linewidth=2, markersize=7)
ax.fill_between(range(len(vc)), vc.values, alpha=0.15, color=IND)
ax.set_xticks(range(len(vc))); ax.set_xticklabels(lbls_v, rotation=20, fontsize=9)
ax.set_title('Churn by Vintage (days)', fontweight='bold'); ax.set_ylabel('Churn Rate (%)')

plt.tight_layout()
plt.savefig(f'{OUT}/eda_overview.png', bbox_inches='tight', dpi=150)
plt.close(); print("[✓] eda_overview.png")

# ── 8b. Feature distributions: churn=0 vs churn=1 ────────────────────────────
# Key numeric features shown separately for stayed vs churned
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Feature Distributions — Stayed vs Churned',
             fontsize=14, fontweight='bold', color=DARK, y=1.01)

features_dist = [
    ('current_balance',   'Current Balance (clipped at 50K)'),
    ('balance_change',    'Balance Change'),
    ('credit_debit_ratio','Credit/Debit Ratio (clipped at 8)'),
    ('days_since_last_tx','Days Since Last Transaction (clipped 500)'),
    ('avg_balance_trend', 'Avg Balance Trend'),
    ('debit_activity_ratio','Debit Activity Ratio'),
]

clips = {
    'current_balance': (None, 50000), 'balance_change': (-20000, 20000),
    'credit_debit_ratio': (0, 8), 'days_since_last_tx': (0, 500),
    'avg_balance_trend': (-20000, 20000), 'debit_activity_ratio': (0, 5),
}

for (feat, title), ax in zip(features_dist, axes.flatten()):
    lo, hi = clips.get(feat, (None, None))
    for cv_, lbl, c in [(0,'Stayed',TEAL),(1,'Churned',ROSE)]:
        data = df[df['churn']==cv_][feat].clip(lo, hi)
        ax.hist(data, bins=40, alpha=0.6, label=lbl, color=c,
                edgecolor='white', linewidth=0.3, density=True)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    # Median lines
    for cv_, c in [(0,TEAL),(1,ROSE)]:
        med = df[df['churn']==cv_][feat].median()
        ax.axvline(med, color=c, linestyle='--', lw=1.5, alpha=0.8)

plt.tight_layout()
plt.savefig(f'{OUT}/eda_distributions.png', bbox_inches='tight', dpi=150)
plt.close(); print("[✓] eda_distributions.png")

# ── 8c. Churn by balance range ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Churn Analysis by Balance Range & Feature Correlation with Churn',
             fontsize=13, fontweight='bold', color=DARK)

ax = axes[0]
bal_bins  = [df['current_balance'].min()-1, 0, 500, 2000, 5000, 15000, df['current_balance'].max()+1]
bal_lbls  = ['Negative','0-500','500-2K','2K-5K','5K-15K','15K+']
df['bal_range'] = pd.cut(df['current_balance'], bins=bal_bins, labels=bal_lbls)
bal_churn = df.groupby('bal_range', observed=False)['churn'].agg(['mean','count']).reset_index()
bar_colors = [ROSE if v > df['churn'].mean() else TEAL for v in bal_churn['mean']]
bars = ax.bar(bal_churn['bal_range'].astype(str), bal_churn['mean']*100,
              color=bar_colors, edgecolor='white', linewidth=1.5)
for bar, (_, row) in zip(bars, bal_churn.iterrows()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{row["mean"]:.1%}\n(n={row["count"]:,})',
            ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=DARK)
ax.axhline(df['churn'].mean()*100, color='gray', linestyle='--', lw=1.3,
           label=f'Overall: {df["churn"].mean():.1%}')
ax.set_title('Churn Rate by Balance Range', fontweight='bold')
ax.set_ylabel('Churn Rate (%)'); ax.legend(fontsize=9)
ax.set_xticklabels(bal_lbls, rotation=20, ha='right')
ax.set_ylim(0, bal_churn['mean'].max()*130)

# Correlation with churn (numeric features only)
ax = axes[1]
numeric_df = df_clean.select_dtypes(include=[np.number])
corr_churn = numeric_df.corr()['churn'].drop('churn').sort_values()
colors_corr = [ROSE if v > 0 else TEAL for v in corr_churn.values]
ax.barh(corr_churn.index, corr_churn.values, color=colors_corr,
        edgecolor='white', linewidth=0.5)
ax.axvline(0, color='gray', lw=1)
ax.set_title('Feature Correlation with Churn Target', fontweight='bold')
ax.set_xlabel('Pearson Correlation')
for i, (idx, val) in enumerate(corr_churn.items()):
    ha = 'left' if val >= 0 else 'right'
    offset = 0.002 if val >= 0 else -0.002
    ax.text(val+offset, i, f'{val:.3f}', va='center', fontsize=7.5,
            color=DARK, ha=ha)

plt.tight_layout()
plt.savefig(f'{OUT}/eda_balance_corr.png', bbox_inches='tight', dpi=150)
plt.close(); print("[✓] eda_balance_corr.png")


# =============================================================================
# 9. MODEL EVALUATION CHARTS
# =============================================================================
print("\n── 9. EVALUATION CHARTS ──────────────────────────────────────────────")

# ── 9a. ROC + Precision-Recall + Confusion Matrix ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle('Model Evaluation — Test Set', fontsize=14, fontweight='bold', color=DARK)

model_colors_map = {
    'Logistic Regression':   IND,
    'Random Forest (tuned)': TEAL,
    'Gradient Boosting':     ROSE,
}

# ROC Curve
ax = axes[0]
for name, res in test_results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, color=model_colors_map[name], lw=2,
            label=f'{name}  AUC={res["ROCAUC"]:.3f}')
ax.plot([0,1],[0,1],'k--',lw=1.5,alpha=0.5,label='Random baseline')
ax.fill_between([0,1],[0,1],alpha=0.04,color='gray')
ax.set_title('ROC Curve', fontweight='bold')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=8.5); ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

# Precision-Recall Curve (important for imbalanced classes!)
# A high area under PR curve = model works well even on minority class.
ax = axes[1]
for name, res in test_results.items():
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    ap = res['AvgPrec']
    ax.plot(rec, prec, color=model_colors_map[name], lw=2,
            label=f'{name}  AP={ap:.3f}')
baseline_pr = y_test.mean()
ax.axhline(baseline_pr, color='gray', linestyle='--', lw=1.5,
           label=f'Baseline (random) = {baseline_pr:.3f}')
ax.set_title('Precision-Recall Curve\n(Critical for imbalanced classes)', fontweight='bold')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.legend(fontsize=8.5); ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

# Confusion Matrix — best AUC model
ax = axes[2]
best_res = test_results[BEST]
cm = confusion_matrix(y_test, best_res['y_pred'])
# TN FP  |  FN (missed churners) = most costly for bank (lost revenue)
# FN TP  |  FP (false alarms)    = wasted outreach budget
im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
ax.set_title(f'Confusion Matrix\n({BEST})', fontweight='bold')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Pred: Stay','Pred: Churn'])
ax.set_yticklabels(['Actual: Stay','Actual: Churn'])
thresh = cm.max()/2
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                fontsize=12, fontweight='bold',
                color='white' if cm[i,j]>thresh else DARK)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f'{OUT}/model_evaluation.png', bbox_inches='tight', dpi=150)
plt.close(); print("[✓] model_evaluation.png")

# ── 9b. Feature Importance — all 3 models ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Feature Importance — All Three Models', fontsize=14,
             fontweight='bold', color=DARK)

# RF: native tree-based importance
ax = axes[0]
fi_rf = pd.Series(rf_tuned.feature_importances_, index=X.columns).sort_values(ascending=True).tail(12)
c_rf  = [ROSE if v>=fi_rf.quantile(0.75) else TEAL for v in fi_rf.values]
ax.barh(fi_rf.index, fi_rf.values, color=c_rf, edgecolor='white', linewidth=0.5)
ax.set_title('RF (Tuned) — Feature Importance', fontweight='bold')
ax.set_xlabel('Mean Decrease Impurity')
for i,(idx,val) in enumerate(fi_rf.items()):
    ax.text(val+0.001, i, f'{val:.3f}', va='center', fontsize=8, color=SLATE)

# GB: native feature importance
ax = axes[1]
fi_gb = pd.Series(gb_model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(12)
c_gb  = [ROSE if v>=fi_gb.quantile(0.75) else AMBER for v in fi_gb.values]
ax.barh(fi_gb.index, fi_gb.values, color=c_gb, edgecolor='white', linewidth=0.5)
ax.set_title('Gradient Boosting — Feature Importance', fontweight='bold')
ax.set_xlabel('Mean Decrease Impurity')
for i,(idx,val) in enumerate(fi_gb.items()):
    ax.text(val+0.001, i, f'{val:.3f}', va='center', fontsize=8, color=SLATE)

# LR: absolute coefficients (after scaling → comparable magnitudes)
ax = axes[2]
lr_inner = lr_pipe.named_steps['lr']
coefs = pd.Series(np.abs(lr_inner.coef_[0]), index=X.columns).sort_values(ascending=True).tail(12)
c_lr  = [ROSE if v>=coefs.quantile(0.75) else IND for v in coefs.values]
ax.barh(coefs.index, coefs.values, color=c_lr, edgecolor='white', linewidth=0.5)
ax.set_title('Logistic Regression — |Coefficient|', fontweight='bold')
ax.set_xlabel('|Coefficient| (scaled features)')
for i,(idx,val) in enumerate(coefs.items()):
    ax.text(val+0.001, i, f'{val:.3f}', va='center', fontsize=8, color=SLATE)

plt.tight_layout()
plt.savefig(f'{OUT}/feature_importance_all.png', bbox_inches='tight', dpi=150)
plt.close(); print("[✓] feature_importance_all.png")

# ── 9c. Predicted probability distributions ───────────────────────────────────
# Shows how confidently the model separates churners from non-churners.
# Good model = two clearly separated humps; bad model = overlapping distributions.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle('Predicted Probability Distributions  P(churn=1)',
             fontsize=13, fontweight='bold', color=DARK)

for ax, (name, res) in zip(axes, test_results.items()):
    for cv_, lbl, c in [(0,'Stayed',TEAL),(1,'Churned',ROSE)]:
        mask = y_test == cv_
        ax.hist(res['y_prob'][mask], bins=40, alpha=0.6, label=lbl,
                color=c, edgecolor='white', linewidth=0.3, density=True)
    ax.axvline(0.5, color='gray', linestyle='--', lw=1.5, label='Threshold 0.5')
    ax.set_title(name, fontweight='bold')
    ax.set_xlabel('P(churn)'); ax.set_ylabel('Density')
    ax.legend(fontsize=8.5)

plt.tight_layout()
plt.savefig(f'{OUT}/prob_distributions.png', bbox_inches='tight', dpi=150)
plt.close(); print("[✓] prob_distributions.png")

# ── 9d. Key insights chart ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Key Insights', fontsize=14, fontweight='bold', color=DARK)

# Age churn
ax = axes[0]
age_bins=[0,25,35,45,55,65,100]; age_lbls=['<25','25-35','35-45','45-55','55-65','65+']
df['age_range'] = pd.cut(df['age'], bins=age_bins, labels=age_lbls)
ac = df.groupby('age_range',observed=False)['churn'].mean()*100
avg_c = df['churn'].mean()*100
bc = [ROSE if v>avg_c else TEAL for v in ac.values]
bars = ax.bar(ac.index.astype(str), ac.values, color=bc, edgecolor='white')
for bar, val in zip(bars, ac.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold', color=DARK)
ax.axhline(avg_c, color='gray', linestyle='--', lw=1.3, label=f'Avg: {avg_c:.1f}%')
ax.set_title('Churn by Age Group', fontweight='bold'); ax.legend(fontsize=9)
ax.set_ylabel('Churn Rate (%)')

# Occ x Gender
ax = axes[1]
pivot = df.groupby(['occupation','gender'])['churn'].mean().unstack()*100
x = np.arange(len(pivot.index))
for i,(g,c) in enumerate(zip(pivot.columns,[ROSE,TEAL])):
    ax.bar(x+(i-0.5)*0.35, pivot[g], 0.35, label=g, color=c, edgecolor='white', alpha=0.9)
ax.set_title('Churn: Occupation × Gender', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(pivot.index, rotation=15, fontsize=9)
ax.set_ylabel('Churn Rate (%)'); ax.legend(fontsize=9)

# Days since tx boxplot
ax = axes[2]
d0 = df[df['churn']==0]['days_since_last_tx'].clip(0,600)
d1 = df[df['churn']==1]['days_since_last_tx'].clip(0,600)
bp = ax.boxplot([d0,d1], patch_artist=True, notch=True,
                medianprops={'color':'white','linewidth':2})
bp['boxes'][0].set_facecolor(TEAL+'99'); bp['boxes'][1].set_facecolor(ROSE+'99')
for w in bp['whiskers']: w.set(color=SLATE,linewidth=1.2)
for c in bp['caps']:     c.set(color=SLATE,linewidth=1.2)
for f in bp['fliers']:   f.set(marker='.',alpha=0.3,color=SLATE,markersize=3)
ax.set_xticklabels(['Stayed','Churned'])
ax.set_title('Days Since Last Transaction', fontweight='bold'); ax.set_ylabel('Days')
ax.text(1.35, d0.median(), f'Med: {d0.median():.0f}d', va='center', fontsize=8, color=TEAL)
ax.text(2.05, d1.median(), f'Med: {d1.median():.0f}d', va='center', fontsize=8, color=ROSE)

plt.tight_layout()
plt.savefig(f'{OUT}/key_insights.png', bbox_inches='tight', dpi=150)
plt.close(); print("[✓] key_insights.png")

# Correlation matrix
fig, ax = plt.subplots(figsize=(14, 11))
corr = df_clean.select_dtypes(include=[np.number]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, cmap='RdYlGn', center=0,
            annot=True, fmt='.2f', annot_kws={'size':6},
            linewidths=0.4, linecolor='white')
ax.set_title('Correlation Matrix', fontsize=13, fontweight='bold', color=DARK, pad=15)
plt.tight_layout()
plt.savefig(f'{OUT}/correlation_matrix.png', bbox_inches='tight', dpi=150)
plt.close(); print("[✓] correlation_matrix.png")


# =============================================================================
# 10. BUSINESS ANALYSIS
# =============================================================================
print("\n── 10. BUSINESS ANALYSIS ─────────────────────────────────────────────")

# Use best model on FULL dataset for business segmentation
all_probs = test_results[BEST]['y_prob']
test_df   = X_test.copy()
test_df['actual_churn'] = y_test.values
test_df['churn_prob']   = all_probs

# ── 10a. Risk segmentation ────────────────────────────────────────────────────
# Four tiers based on predicted probability of churn:
#   Low      0.0 – 0.25   → standard service, no intervention
#   Medium   0.25 – 0.50  → soft touch (newsletter, product recommendation)
#   High     0.50 – 0.75  → proactive outreach (call, offer)
#   Critical 0.75 – 1.00  → urgent retention (manager call, discount)
bins_risk  = [0.0, 0.25, 0.50, 0.75, 1.01]
labels_risk= ['Low', 'Medium', 'High', 'Critical']
test_df['risk_segment'] = pd.cut(test_df['churn_prob'],
                                  bins=bins_risk, labels=labels_risk)

risk_summary = test_df.groupby('risk_segment', observed=False).agg(
    clients     = ('churn_prob', 'count'),
    actual_churn= ('actual_churn', 'sum'),
    avg_prob    = ('churn_prob', 'mean'),
).reset_index()
risk_summary['actual_churn_rate'] = (risk_summary['actual_churn'] /
                                      risk_summary['clients'] * 100)

print("\n  Risk Segmentation (test set):")
print(f"  {'Segment':<10} {'Clients':>8} {'Churned':>9} {'Churn Rate':>12} {'Avg P(churn)':>13}")
print("  " + "─"*56)
for _, row in risk_summary.iterrows():
    print(f"  {row['risk_segment']:<10} {row['clients']:>8,} {row['actual_churn']:>9.0f}"
          f" {row['actual_churn_rate']:>11.1f}%  {row['avg_prob']:>12.3f}")

# ── 10b. Retention simulation ─────────────────────────────────────────────────
# Q: If we retain 20% of High+Critical risk clients, how much revenue is saved?
#
# Assumptions:
#   Average monthly revenue per client (CLV proxy) = 500 currency units
#   Retention cost per contacted client             = 50 currency units
#   Retention rate achieved through outreach         = 20%

MONTHLY_REVENUE   = 500    # avg monthly revenue per client retained
RETENTION_COST    = 50     # cost of one outreach attempt
RETENTION_RATE    = 0.20   # fraction of contacted clients successfully retained

high_critical = risk_summary[risk_summary['risk_segment'].isin(['High','Critical'])]
n_targeted    = int(high_critical['clients'].sum())
n_churners_in = int(high_critical['actual_churn'].sum())
n_saved       = int(n_churners_in * RETENTION_RATE)
revenue_saved = n_saved * MONTHLY_REVENUE * 12   # annualised
outreach_cost = n_targeted * RETENTION_COST
net_benefit   = revenue_saved - outreach_cost

print(f"\n  Retention Simulation (High + Critical segments):")
print(f"    Clients targeted         : {n_targeted:,}")
print(f"    Actual churners targeted : {n_churners_in:,}")
print(f"    Retained @ {RETENTION_RATE:.0%}           : {n_saved:,} clients")
print(f"    Annual revenue saved     : {revenue_saved:,}")
print(f"    Outreach cost            : {outreach_cost:,}")
print(f"    Net benefit (annual)     : {net_benefit:,}")

# ── 10c. Error cost analysis ──────────────────────────────────────────────────
# FP = False Positive: model says "churn", client stays → wasted outreach cost
# FN = False Negative: model says "stay", client leaves → lost annual revenue
COST_FP = RETENTION_COST            # wasted outreach (one contact)
COST_FN = MONTHLY_REVENUE * 12      # lost annual CLV

cm_best = confusion_matrix(y_test, test_results[BEST]['y_pred'])
tn, fp, fn, tp = cm_best.ravel()
total_fp_cost = fp * COST_FP
total_fn_cost = fn * COST_FN
total_cost    = total_fp_cost + total_fn_cost

print(f"\n  Error Cost Analysis ({BEST}):")
print(f"    False Positives  : {fp:,}  × {COST_FP} = {total_fp_cost:,}")
print(f"    False Negatives  : {fn:,}  × {COST_FN:,} = {total_fn_cost:,}")
print(f"    Total model cost : {total_cost:,}")

# ── 10d. Business charts ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle('Business Analysis — Risk Segmentation & Retention Value',
             fontsize=13, fontweight='bold', color=DARK)

seg_colors = {'Low': TEAL, 'Medium': AMBER, 'High': ROSE, 'Critical': '#7C0A02'}

# Risk distribution — clients per segment
ax = axes[0]
bars = ax.bar(risk_summary['risk_segment'].astype(str),
              risk_summary['clients'],
              color=[seg_colors[s] for s in risk_summary['risk_segment']],
              edgecolor='white', linewidth=1.5)
for bar, (_, row) in zip(bars, risk_summary.iterrows()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
            f'{row["clients"]:,}\n({row["clients"]/len(test_df):.1%})',
            ha='center', fontsize=9, fontweight='bold', color=DARK)
ax.set_title('Clients by Risk Segment', fontweight='bold')
ax.set_ylabel('Number of Clients'); ax.set_ylim(0, risk_summary['clients'].max()*1.25)

# Actual churn rate per segment
ax = axes[1]
bars = ax.bar(risk_summary['risk_segment'].astype(str),
              risk_summary['actual_churn_rate'],
              color=[seg_colors[s] for s in risk_summary['risk_segment']],
              edgecolor='white', linewidth=1.5)
for bar, (_, row) in zip(bars, risk_summary.iterrows()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
            f'{row["actual_churn_rate"]:.1f}%',
            ha='center', fontsize=10, fontweight='bold', color=DARK)
ax.set_title('Actual Churn Rate by Segment', fontweight='bold')
ax.set_ylabel('Churn Rate (%)'); ax.set_ylim(0, 110)

# Retention economics — waterfall-style
ax = axes[2]
categories = ['Revenue\nSaved', 'Outreach\nCost', 'Net\nBenefit']
values      = [revenue_saved, -outreach_cost, net_benefit]
colors_ret  = [TEAL, ROSE, GREEN if net_benefit > 0 else ROSE]
bars = ax.bar(categories, [abs(v) for v in values], color=colors_ret,
              edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+500,
            f'{abs(val):,}', ha='center', fontsize=10, fontweight='bold', color=DARK)
ax.set_title(f'Retention Economics\n(High+Critical, {RETENTION_RATE:.0%} retention rate)',
             fontweight='bold')
ax.set_ylabel('Currency Units')

plt.tight_layout()
plt.savefig(f'{OUT}/business_analysis.png', bbox_inches='tight', dpi=150)
plt.close(); print("[✓] business_analysis.png")


# =============================================================================
# 11. CV RESULTS CHART
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('5-Fold Cross-Validation Results (Train Set)',
             fontsize=13, fontweight='bold', color=DARK)

metrics_to_plot = ['roc_auc', 'f1', 'recall', 'precision']
metric_labels   = ['ROC-AUC', 'F1-Score', 'Recall', 'Precision']

# Box plots of CV scores
ax = axes[0]
plot_data  = []
plot_labels= []
plot_colors= []
for name, scores in cv_results.items():
    for m in metrics_to_plot:
        plot_data.append(scores[m])
        plot_labels.append(f'{name[:6]}\n{m[:5]}')
        plot_colors.append(list(model_colors_map.values())[list(cv_results.keys()).index(name)])

# Simplified: group by model, show ROC-AUC box
positions = np.arange(len(cv_results))
for i, (name, scores) in enumerate(cv_results.items()):
    c = list(model_colors_map.values())[i]
    bp = ax.boxplot(scores['roc_auc'], positions=[i], patch_artist=True,
                    medianprops={'color':'white','linewidth':2},
                    widths=0.5)
    bp['boxes'][0].set_facecolor(c+'99')
    for w in bp['whiskers']: w.set(color=SLATE, linewidth=1.2)
    for cap in bp['caps']:   cap.set(color=SLATE, linewidth=1.2)
ax.set_xticks(positions)
ax.set_xticklabels([n[:20] for n in cv_results.keys()], rotation=10, fontsize=9)
ax.set_title('ROC-AUC Distribution across 5 Folds', fontweight='bold')
ax.set_ylabel('ROC-AUC')

# Mean metrics comparison bar chart
ax = axes[1]
x  = np.arange(len(metrics_to_plot))
w  = 0.25
for i, (name, scores) in enumerate(cv_results.items()):
    means = [scores[m].mean() for m in metrics_to_plot]
    c     = list(model_colors_map.values())[i]
    bars_cv = ax.bar(x + i*w - w, means, w, label=name[:20],
                     color=c, edgecolor='white', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=10)
ax.set_title('Mean CV Metrics by Model', fontweight='bold')
ax.set_ylabel('Score'); ax.legend(fontsize=9)
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig(f'{OUT}/cv_results.png', bbox_inches='tight', dpi=150)
plt.close(); print("[✓] cv_results.png")


# =============================================================================
# 12. FINAL SUMMARY PRINT
# =============================================================================
print("\n" + "=" * 70)
print("  FINAL RESULTS")
print("=" * 70)

print(f"\n{'Metric':<14}", end='')
for name in test_results: print(f"  {name[:22]:>22}", end='')
print(); print("─"*82)
for label, key in [('Accuracy','Accuracy'),('Precision','Precision'),
                   ('Recall','Recall'),('F1','F1'),('ROC-AUC','ROCAUC'),('Avg Prec','AvgPrec')]:
    print(f"{label:<14}", end='')
    for res in test_results.values(): print(f"  {res[key]:>22.4f}", end='')
    print()

print(f"\n5-Fold CV Summary (train set):")
for name, scores in cv_results.items():
    print(f"  {name:<25}  AUC={scores['roc_auc'].mean():.3f}±{scores['roc_auc'].std():.3f}"
          f"  F1={scores['f1'].mean():.3f}±{scores['f1'].std():.3f}")

print(f"\nBest tuned RF params: {rscv.best_params_}")
print(f"Best RF CV AUC: {rscv.best_score_:.4f}")
print(f"RF Validation AUC: {rf_val_auc:.4f}")

top5_rf = pd.Series(rf_tuned.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
print(f"\nTop-5 features (RF):")
for feat, imp in top5_rf.items():
    print(f"  {feat:<40} {imp:.4f}")

print(f"\nBusiness (test set, High+Critical):")
print(f"  Clients at risk   : {n_targeted:,}")
print(f"  Net annual benefit: {net_benefit:,} (at {RETENTION_RATE:.0%} retention)")

print(f"\nOutput images:")
for img in ['eda_overview','eda_distributions','eda_balance_corr','correlation_matrix',
            'model_evaluation','feature_importance_all','prob_distributions',
            'cv_results','key_insights','business_analysis']:
    print(f"  {img}.png")
print("=" * 70)
