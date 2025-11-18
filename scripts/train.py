import pickle
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV


# ---------- PATHS (robust) ----------
# This file: mlzoomcamp-heart/scripts/train.py
BASE_DIR = Path(__file__).resolve().parent.parent   # -> mlzoomcamp-heart/

DATA_PATH = BASE_DIR / "data" / "Heart_disease_cleveland_new.csv"
MODEL_PATH = BASE_DIR / "models" / "model.bin"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def train_and_evaluate(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=0,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- Logistic Regression ----
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    y_pred_lr = lr.predict(X_test_scaled)
    y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

    acc_lr = accuracy_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_proba_lr)

    print(f"Logistic Regression - accuracy: {acc_lr:.3f}")
    print(f"Logistic Regression - ROC AUC: {auc_lr:.3f}")

    # ---- Random Forest (baseline) ----
    rf = RandomForestClassifier(n_estimators=200, random_state=0)
    rf.fit(X_train_scaled, y_train)

    y_pred_rf = rf.predict(X_test_scaled)
    y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_proba_rf)

    print(f"Random Forest - accuracy: {acc_rf:.3f}")
    print(f"Random Forest - ROC AUC: {auc_rf:.3f}")

    # ---- Random Forest tuning ----
    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 3, 5, 10],
        "min_samples_split": [2, 5, 10],
    }

    rf_base = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    grid_search.fit(X_train_scaled, y_train)

    print("Best params:", grid_search.best_params_)
    print("Best CV ROC AUC:", grid_search.best_score_)

    best_rf = grid_search.best_estimator_

    y_pred_best = best_rf.predict(X_test_scaled)
    y_proba_best = best_rf.predict_proba(X_test_scaled)[:, 1]

    acc_best = accuracy_score(y_test, y_pred_best)
    auc_best = roc_auc_score(y_test, y_proba_best)

    print(f"Tuned RF - accuracy: {acc_best:.3f}")
    print(f"Tuned RF - ROC AUC: {auc_best:.3f}")

    return scaler, best_rf


def save_model(scaler, model, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f_out:
        pickle.dump((scaler, model), f_out)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    df = load_data()
    scaler, model = train_and_evaluate(df)
    save_model(scaler, model)
