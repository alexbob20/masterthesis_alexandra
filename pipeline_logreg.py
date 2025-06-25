
from sklearn.base import clone

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve
)

import pandas as pd


import pandas as pd
df = pd.read_csv("excel_files/baseline_features.csv",na_values=["#NULL!", "NULL", "N/A"], decimal=",")

df = df.apply(pd.to_numeric, errors="coerce")
df = df[df.isnull().sum(axis=1) <= 500]



var_thresh = VarianceThreshold(threshold=0.0)
scaler = RobustScaler()
kbest = SelectKBest(score_func=mutual_info_classif, k=100)
pca = PCA(n_components=50)

scoring = {
    'roc_auc': 'roc_auc',
    'pr_auc': 'average_precision',
    'f1': 'f1',
    'balanced_accuracy': 'balanced_accuracy'
}

models = {
        "No FS - Balanced": Pipeline([
            ('var', var_thresh),
            ('scaler', scaler),
            ('logreg', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ]),
        "No FS - SMOTE": ImbPipeline([
            ('var', var_thresh),
            ('scaler', scaler),
            ('smote', FunctionSampler(func=safe_smote, validate=False)),
            ('logreg', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "No FS - ElasticNet": Pipeline([
            ('var', var_thresh),
            ('scaler', scaler),
            ('logreg', LogisticRegressionCV(
                Cs=10,
                penalty='elasticnet',
                solver='saga',
                l1_ratios=[0.5],
                class_weight='balanced',
                cv=5,
                scoring='average_precision',
                max_iter=2000,
                random_state=42
            ))
        ]),

        "KBest - Balanced": Pipeline([
            ('var', var_thresh),
            ('scaler', scaler),
            ('kbest', kbest),
            ('logreg', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ]),
        "KBest - SMOTE": ImbPipeline([
            ('var', var_thresh),
            ('scaler', scaler),
            ('kbest', kbest),
            ('smote', FunctionSampler(func=safe_smote, validate=False)),
            ('logreg', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "KBest - ElasticNet": Pipeline([
            ('var', var_thresh),
            ('scaler', scaler),
            ('kbest', kbest),
            ('logreg', LogisticRegressionCV(
                Cs=10,
                penalty='elasticnet',
                solver='saga',
                l1_ratios=[0.5],
                class_weight='balanced',
                cv=5,
                scoring='average_precision',
                max_iter=2000,
                random_state=42
            ))
        ]),

        "PCA - Balanced": Pipeline([
            ('var', var_thresh),
            ('scaler', scaler),
            ('pca', pca),
            ('logreg', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ]),
        "PCA - SMOTE": ImbPipeline([
            ('var', var_thresh),
            ('scaler', scaler),
            ('pca', pca),
            ('smote', FunctionSampler(func=safe_smote, validate=False)),
            ('logreg', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "PCA - ElasticNet": Pipeline([
            ('var', var_thresh),
            ('scaler', scaler),
            ('pca', pca),
            ('logreg', LogisticRegressionCV(
                Cs=10,
                penalty='elasticnet',
                solver='saga',
                l1_ratios=[0.5],
                class_weight='balanced',
                cv=5,
                scoring='average_precision',
                max_iter=2000,
                random_state=42
            ))
        ])
    }
# Re-import necessary libraries after reset
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve, f1_score, precision_score,
    recall_score, balanced_accuracy_score
)
from sklearn.base import clone

from joblib import Parallel, delayed

def evaluate_fold(model, X_train_fold, y_train_fold, X_val_fold, y_val_fold, min_precision, min_recall, recall_target):
    model_fold = clone(model)
    model_fold.fit(X_train_fold, y_train_fold)
    y_proba_val = model_fold.predict_proba(X_val_fold)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_val_fold, y_proba_val)
    f1s = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precisions, recalls)]
    valid_idxs = [i for i, (p, r) in enumerate(zip(precisions[:-1], recalls[:-1])) if p >= min_precision and r >= min_recall]

    if valid_idxs:
        best_idx = valid_idxs[np.argmax([f1s[i] for i in valid_idxs])]
        threshold = thresholds[best_idx]
    else:
        threshold = 0.5

    y_pred_val = (y_proba_val >= threshold).astype(int)
    f1 = f1_score(y_val_fold, y_pred_val, zero_division=0)
    precision = precision_score(y_val_fold, y_pred_val, zero_division=0)
    recall = recall_score(y_val_fold, y_pred_val, zero_division=0)
    bal_acc = balanced_accuracy_score(y_val_fold, y_pred_val)
    prec_at_70 = precision if recall >= recall_target else np.nan

    return f1, prec_at_70, threshold, bal_acc

# Updated cross_val_with_threshold_tuning with n_jobs
def cross_val_with_threshold_tuning(model, X, y, min_precision=0.3, min_recall=0.3, recall_target=0.7, n_splits=5, n_jobs=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_fold)(
            model,
            X.iloc[train_idx], y.iloc[train_idx],
            X.iloc[val_idx], y.iloc[val_idx],
            min_precision, min_recall, recall_target
        )
        for train_idx, val_idx in kf.split(X, y)
    )

    f1_scores, precisions_at_70recall, thresholds_used, bal_accs = zip(*results)

    return {
        "F1 Mean": np.mean(f1_scores),
        "F1 Std": np.std(f1_scores),
        "Balanced Accuracy Mean": np.mean(bal_accs),
        "Balanced Accuracy Std": np.std(bal_accs),
        "Precision@70%Recall Mean": np.nanmean(precisions_at_70recall),
        "Thresholds Used (Mean)": np.mean(thresholds_used)
    }

from imblearn.over_sampling import SMOTE

def safe_smote(X, y, random_state=42):
    # Count minority class samples
    from collections import Counter
    counter = Counter(y)
    min_class_count = min(counter.values())

    # Avoid errors by capping k_neighbors
    k = min(3, min_class_count - 1) if min_class_count > 1 else 1

    return SMOTE(k_neighbors=k, random_state=random_state).fit_resample(X, y)

cv_results = []

for target in tqdm(targets_5050):
    data = df[[target]].join(df[df.columns[90:]]).dropna()
    y = data[target]
    X = data[df.columns[90:]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    for model_name, model in models.items():
        metrics = cross_val_with_threshold_tuning(
            model, X_train, y_train,  # <- important: use training set only
            min_precision=0.3,
            min_recall=0.3,
            recall_target=0.7,
            n_splits=5
        )

        cv_results.append({
            "Target": target,
            "Model": model_name,
            "F1 Mean": metrics["F1 Mean"],
            "F1 Std": metrics["F1 Std"],
            "Balanced Accuracy Mean": metrics["Balanced Accuracy Mean"],
            "Balanced Accuracy Std": metrics["Balanced Accuracy Std"],
            "Precision@70%Recall Mean": metrics["Precision@70%Recall Mean"],
            "Thresholds Used (Mean)": metrics["Thresholds Used (Mean)"]
        })

# Final results per model and target
cv_df_5050 = pd.DataFrame(cv_results)

# Example: find best model per target
best_models_df_5050 = cv_df_5050.sort_values(['Target', 'F1 Mean'], ascending=[True, False]).groupby('Target').first().reset_index()

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix, precision_recall_curve
)

final_results = []

for _, row in best_models.iterrows():
    target = row["Target"]
    model_name = row["Model"]
    model = clone(models[model_name])

    # Prepare data
    data = df[[target]].join(df[df.columns[90:]]).dropna()
    y = data[target]
    X = data[df.columns[90:]]

    # Train/test split (held-out test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Fit on train set
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ----- Precision-Recall Curve & Threshold Tuning -----
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precisions, recalls)]
    valid_idxs = [
        i for i, (p, r) in enumerate(zip(precisions[:-1], recalls[:-1]))
        if p >= 0.3 and r >= 0.3
    ]

    if valid_idxs:
        best_f1_idx = valid_idxs[np.argmax([f1_scores[i] for i in valid_idxs])]
        threshold = thresholds[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]
    else:
        threshold = 0.5  # fallback
        best_f1 = np.nan

    # Final prediction using selected threshold
    y_pred = (y_proba >= threshold).astype(int)
    prec_at_thresh = precision_score(y_test, y_pred, zero_division=0)
    recall_at_thresh = recall_score(y_test, y_pred, zero_division=0)
    # Compute precision@70% recall from the full PR curve
    recall_target = 0.7
    valid_idxs_70 = [i for i, r in enumerate(recalls[:-1]) if r >= recall_target]
    if valid_idxs_70:
        precision_at_70recall = max([precisions[i] for i in valid_idxs_70])
    else:
        precision_at_70recall = np.nan


    # Confusion matrix and metrics
    acc = accuracy_score(y_test, y_pred)
    precision = prec_at_thresh
    recall = recall_at_thresh
    f1 = f1_score(y_test, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    final_results.append({
        "Target": target,
        "Model": model_name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 (test)": f1,
        "Balanced Accuracy (test)": bal_acc,
        "Threshold Used": threshold,
        "F1 at Threshold": best_f1,
        "Precision@70%Recall": precision_at_70recall,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    })

# Convert to DataFrame
final_results_df_5050 = pd.DataFrame(final_results)
def extract_top_features_no_pca(df, best_models_df, models, top_n=3):
    results = []
    acoustic_features = df.columns[90:]  # Adjust if needed

    for _, row in best_models_df.iterrows():
        target = row["Target"]
        model_name = row["Model"]

        if "PCA" in model_name:
            continue  # Skip PCA models

        # Prepare data
        data = df[[target]].join(df[acoustic_features]).dropna()
        X = data[acoustic_features]
        y = data[target]

        # Fit the best model
        model = clone(models[model_name])
        model.fit(X, y)

        # Get final logistic regression step
        if hasattr(model, "named_steps"):
            logreg = model.named_steps.get("logreg")
            if logreg is None:
                continue
        else:
            continue  # Not a pipeline

        # Get feature names after kbest (if present)
        if "kbest" in model.named_steps:
            kbest = model.named_steps["kbest"]
            selected_idxs = kbest.get_support(indices=True)
            feature_names = acoustic_features[selected_idxs]
        else:
            feature_names = acoustic_features

        # Extract coefficients
        coefs = logreg.coef_[0]
        top_idx = np.argsort(np.abs(coefs))[-top_n:][::-1]
        top_features = [(feature_names[i], coefs[i]) for i in top_idx]

        for feat, weight in top_features:
            results.append({
                "Target": target,
                "Model": model_name,
                "Feature": feat,
                "Coefficient": weight
            })

    return pd.DataFrame(results)

# Example usage:
top_features_df = extract_top_features_no_pca(df, final_results_df_5050, models)
display(top_features_df)
