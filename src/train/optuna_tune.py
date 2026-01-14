import os
import argparse
import pandas as pd
import numpy as np
import joblib
import optuna
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data/raw/breast_cancer.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    target_col = "Diagnosis" if "Diagnosis" in df.columns else df.columns[-1]
    y = df[target_col].astype(str).map({"B": 0, "M": 1})
    X = df.drop(columns=[target_col])
    return X, y

def make_model(model_type: str, params: dict):
    if model_type == "rf":
        return RandomForestClassifier(**params)
    elif model_type == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, **params))
        ])
    else:
        raise ValueError("model_type must be rf or svm")

def cv_metrics(model, X, y, seed=42):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # cross_val_predict pour récupérer proba + prédictions
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)

    f1 = f1_score(y, pred)
    auc = roc_auc_score(y, proba)
    return f1, auc

def main(model_type="rf", n_trials=10, seed=42):
    X, y = load_data()

    # On garde un vrai test set final (jamais touché par Optuna)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    mlflow.set_experiment("breast_cancer_ucimlrepo")

    with mlflow.start_run(run_name=f"optuna_{model_type}_study") as parent_run:
        mlflow.log_param("study_model_type", model_type)
        mlflow.log_param("n_trials", n_trials)

        def objective(trial: optuna.Trial):
            if model_type == "rf":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
                    "random_state": seed,
                }
            else:  # svm
                params = {
                    "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
                    "gamma": trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
                    "kernel": "rbf",
                }

            model = make_model(model_type, params)

            # Run MLflow pour chaque trial (nested)
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                for k, v in params.items():
                    mlflow.log_param(k, v)

                f1, auc = cv_metrics(model, X_train, y_train, seed=seed)
                mlflow.log_metric("cv_f1", f1)
                mlflow.log_metric("cv_roc_auc", auc)

            # On optimise surtout F1 (classification binaire santé)
            return f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_f1", study.best_value)

        # Entraîner modèle final sur train complet, évaluer sur test
        final_model = make_model(model_type, {
            **best_params,
            **({"random_state": seed} if model_type == "rf" else {})
        })
        final_model.fit(X_train, y_train)

        proba_test = final_model.predict_proba(X_test)[:, 1]
        pred_test = (proba_test >= 0.5).astype(int)

        test_f1 = f1_score(y_test, pred_test)
        test_auc = roc_auc_score(y_test, proba_test)

        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_roc_auc", test_auc)

        os.makedirs("models", exist_ok=True)
        out_path = f"models/optuna_best_{model_type}.joblib"
        joblib.dump(final_model, out_path)
        mlflow.log_artifact(out_path, artifact_path="model_files")
        mlflow.sklearn.log_model(final_model, artifact_path="sk_model")

        print("✅ BEST PARAMS:", best_params)
        print(f"✅ TEST: f1={test_f1:.4f} auc={test_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "svm"], default="rf")
    parser.add_argument("--n_trials", type=int, default=10)
    args = parser.parse_args()
    main(model_type=args.model, n_trials=args.n_trials)
