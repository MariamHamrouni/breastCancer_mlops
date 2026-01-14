import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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

def train_rf(X_train, y_train, n_estimators=200, max_depth=None, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, C=1.0, kernel="rbf", gamma="scale"):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=C, kernel=kernel, gamma=gamma, probability=True))
    ])
    model.fit(X_train, y_train)
    return model

def eval_and_log(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", auc)

    return acc, f1, auc

def main(model_type="rf"):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("breast_cancer_ucimlrepo")

    with mlflow.start_run(run_name=f"{model_type}_baseline"):
        mlflow.log_param("model_type", model_type)

        if model_type == "rf":
            model = train_rf(X_train, y_train, n_estimators=200)
            mlflow.log_param("n_estimators", 200)
        elif model_type == "svm":
            model = train_svm(X_train, y_train, C=1.0, kernel="rbf")
            mlflow.log_param("C", 1.0)
            mlflow.log_param("kernel", "rbf")
        else:
            raise ValueError("model_type must be 'rf' or 'svm'")

        acc, f1, auc = eval_and_log(model, X_test, y_test)

        os.makedirs("models", exist_ok=True)
        path = f"models/{model_type}_baseline.joblib"
        joblib.dump(model, path)

        mlflow.log_artifact(path, artifact_path="model_files")
        mlflow.sklearn.log_model(model, artifact_path="sk_model")

        print(f"✅ Done {model_type} | acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

if __name__ == "__main__":
    main("rf")
    main("svm")
