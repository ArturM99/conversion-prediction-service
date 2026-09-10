import os
import time
import dill
import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from pipeline import build_pipeline, load_data

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ga_model.pkl")


def train():
    print("НАЧАЛО ОБУЧЕНИЯ")
    df = load_data()
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print()
    print(f"Train: {X_train.shape}")
    print(f"Test:  {X_test.shape}")
    print(f"Train target: {y_train.mean():.4f}")
    print(f"Test target:  {y_test.mean():.4f}")

    pipeline = build_pipeline()

    print()
    print("Обучение Decision Tree...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"Время обучения: {elapsed:.1f} сек")

    train_proba = pipeline.predict_proba(X_train)[:, 1]
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba)
    test_auc = roc_auc_score(y_test, test_proba)

    print()
    print(f"Train ROC-AUC: {train_auc:.4f}")
    print(f"Test ROC-AUC:  {test_auc:.4f}")
    print(f"ROC-AUC gap:   {train_auc - test_auc:.4f}")

    metadata = {
        "name": "Conversion Prediction Model",
        "author": 'Artur',
        "version": 1,
        "date": datetime.datetime.now()}

    model_data = {"model": pipeline, "metadata": metadata}

    with open(MODEL_FILE, "wb") as file:
        dill.dump(model_data, file)

    print(f"Модель сохранена: {MODEL_FILE}")
    return model_data

if __name__ == "__main__":
    train()