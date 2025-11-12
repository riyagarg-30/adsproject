import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import RandomForestClassifier
    XGB_AVAILABLE = False

import shap

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODEL_PATH = os.path.join(MODEL_DIR, 'credit_model.joblib')

FEATURES = [
    'LIMIT_BAL', 'AGE',
    'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
    'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6',
    'category'
]
TARGET = 'default_payment_next_month'


def _load_sample_data() -> pd.DataFrame:
    sample_csv = os.path.join(DATA_DIR, 'sample_credit_data.csv')
    if not os.path.exists(sample_csv):
        os.makedirs(DATA_DIR, exist_ok=True)
        # Create minimal synthetic dataset if not present
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            'LIMIT_BAL': rng.integers(10000, 500000, n),
            'AGE': rng.integers(21, 75, n),
            'BILL_AMT1': rng.integers(0, 200000, n),
            'BILL_AMT2': rng.integers(0, 200000, n),
            'BILL_AMT3': rng.integers(0, 200000, n),
            'BILL_AMT4': rng.integers(0, 200000, n),
            'BILL_AMT5': rng.integers(0, 200000, n),
            'BILL_AMT6': rng.integers(0, 200000, n),
            'PAY_AMT1': rng.integers(0, 100000, n),
            'PAY_AMT2': rng.integers(0, 100000, n),
            'PAY_AMT3': rng.integers(0, 100000, n),
            'PAY_AMT4': rng.integers(0, 100000, n),
            'PAY_AMT5': rng.integers(0, 100000, n),
            'PAY_AMT6': rng.integers(0, 100000, n),
            'category': rng.choice(['Groceries','Utilities','Entertainment','Travel','Other'], n)
        })
        # Create a synthetic target: higher bills vs pays and younger age -> higher default risk
        risk_score = (
            (df[[f'BILL_AMT{i}' for i in range(1,7)]].sum(axis=1) - df[[f'PAY_AMT{i}' for i in range(1,7)]].sum(axis=1)) / 1e5
            + (40 - df['AGE'])/40
        )
        prob = 1 / (1 + np.exp(-risk_score))
        df[TARGET] = (prob > 0.5).astype(int)
        df.to_csv(sample_csv, index=False)
    else:
        df = pd.read_csv(sample_csv)
    return df


def _build_pipeline(df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    numeric_features = [
        'LIMIT_BAL','AGE',
        'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
        'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'
    ]
    cat_features = ['category']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, cat_features)
        ]
    )

    if XGB_AVAILABLE:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='logloss',
            random_state=42,
            n_jobs=4
        )
    else:
        model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=4)

    clf = Pipeline(steps=[('preprocess', preprocessor), ('model', model)])

    # After fitting, we will expand feature names for SHAP where possible
    return clf, numeric_features + cat_features


def train_and_save() -> Dict[str, Any]:
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = _load_sample_data()
    # Ensure required columns
    missing = [c for c in FEATURES if c not in df.columns and c != TARGET]
    for c in missing:
        if c == 'category':
            df[c] = 'Other'
        else:
            df[c] = 0.0

    X = df[FEATURES]
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf, feat_list = _build_pipeline(df)
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        'auc': float(roc_auc_score(y_test, y_proba)),
        'f1': float(f1_score(y_test, y_pred)),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'default_rate': float(y.mean()),
        'avg_spend': float(df[[f'BILL_AMT{i}' for i in range(1,7)]].sum(axis=1).mean())
    }

    # SHAP feature importance (mean |shap|)
    try:
        # Take a sample to speed up
        sample = X_train.sample(min(200, len(X_train)), random_state=42)
        # Extract preprocessed array for SHAP background
        preprocess = clf.named_steps['preprocess']
        model = clf.named_steps['model']
        X_bg = preprocess.fit_transform(sample)
        explainer = None
        if XGB_AVAILABLE:
            explainer = shap.TreeExplainer(model)
        else:
            # KernelExplainer for non-tree models could be slow; RandomForest is tree-based
            explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_bg)
        if isinstance(shap_values, list):
            shap_arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_arr = shap_values
        mean_abs = np.mean(np.abs(shap_arr), axis=0)
        # get feature names after preprocessing
        ohe = preprocess.named_transformers_['cat'].named_steps['onehot']
        cat_names = list(ohe.get_feature_names_out(['category']))
        num_names = [
            'LIMIT_BAL','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
            'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'
        ]
        feature_names = num_names + cat_names
        # Align lengths
        k = min(len(mean_abs), len(feature_names))
        feature_importances = sorted([
            {'feature': feature_names[i], 'importance': float(mean_abs[i])} for i in range(k)
        ], key=lambda x: x['importance'], reverse=True)
    except Exception:
        feature_importances = []

    payload = {
        'pipeline': clf,
        'feature_list': feat_list,
        'metrics': metrics,
        'feature_importances': feature_importances
    }
    joblib.dump(payload, MODEL_PATH)
    return metrics


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save()


def _load_payload() -> Dict[str, Any]:
    ensure_model()
    return joblib.load(MODEL_PATH)


def get_metrics() -> Dict[str, Any]:
    payload = _load_payload()
    return {
        'metrics': payload.get('metrics', {}),
        'feature_importances': payload.get('feature_importances', [])
    }


def _prepare_input_df(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure all features
    for c in FEATURES:
        if c not in df.columns and c != TARGET:
            df[c] = 'Other' if c == 'category' else 0.0
    return df[FEATURES]


def predict_proba_df(df: pd.DataFrame) -> Tuple[List[float], List[str]]:
    payload = _load_payload()
    clf: Pipeline = payload['pipeline']
    X = _prepare_input_df(df)
    probs = clf.predict_proba(X)[:, 1].tolist()
    return probs, FEATURES


def single_predict_proba(record: Dict[str, Any]) -> float:
    payload = _load_payload()
    clf: Pipeline = payload['pipeline']
    df = pd.DataFrame([record])
    X = _prepare_input_df(df)
    prob = clf.predict_proba(X)[:, 1][0]
    return float(prob)


def forecast_next_months(history: List[float], n_months: int = 6) -> Dict[str, List[float]]:
    # Simple linear extrapolation over months
    y = np.array(history, dtype=float)
    x = np.arange(1, len(y) + 1)
    if len(y) >= 2 and np.any(np.isfinite(y)):
        coeffs = np.polyfit(x, y, 1)
        x_future = np.arange(len(y) + 1, len(y) + n_months + 1)
        y_future = np.polyval(coeffs, x_future)
    else:
        y_future = np.full(n_months, y[-1] if len(y) else 0.0)
    return {
        'history': y.tolist(),
        'future': y_future.astype(float).tolist()
    }
