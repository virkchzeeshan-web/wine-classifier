"""Train and compare machine learning models on the Wine dataset.

Run:  python train.py
Output: results printed to the terminal, plus
  - model.joblib             (best model, ready for predict.py)
  - confusion_matrix.png     (test-set performance of the best model)
"""

import joblib
import matplotlib

matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def main():
    # 1. Load data
    data = load_wine()
    X, y = data.data, data.target
    class_names = [str(c) for c in data.target_names]
    print(f"Samples: {X.shape[0]}, features: {X.shape[1]}, classes: {class_names}\n")

    # 2. Split into train and test sets (stratified keeps class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Define candidate models (scaling matters for LogisticRegression and KNN)
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }

    # 4. Compare models with 5-fold cross-validation on the training set
    print("5-fold cross-validation accuracy (training set):")
    cv_scores = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_scores[name] = scores.mean()
        print(f"  {name:<22} {scores.mean():.3f} (+/- {scores.std():.3f})")

    # 5. Pick the best model and evaluate once on the held-out test set
    best_name = max(cv_scores, key=cv_scores.get)
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print(f"\nBest model: {best_name}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

    # 6. Save confusion matrix and model
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=data.target_names, cmap="Blues"
    )
    plt.title(f"Confusion Matrix: {best_name}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)

    joblib.dump({"model": best_model, "target_names": list(data.target_names),
                 "feature_names": list(data.feature_names)}, "model.joblib")
    print("Saved model.joblib and confusion_matrix.png")


if __name__ == "__main__":
    main()
