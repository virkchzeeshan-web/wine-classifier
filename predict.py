"""Predict the wine class for one sample using the trained model.

Run (after train.py):  python predict.py
Edit the `sample` values below to try your own chemical measurements.
"""

import joblib

bundle = joblib.load("model.joblib")
model = bundle["model"]
target_names = bundle["target_names"]
feature_names = bundle["feature_names"]

# One wine sample, in the same order as feature_names
sample = [13.2, 2.77, 2.51, 18.5, 98.0, 2.6, 2.5, 0.29, 1.77, 5.5, 1.05, 3.2, 1050.0]

pred = model.predict([sample])[0]
print("Features:", dict(zip(feature_names, sample)))
print(f"\nPredicted class: {target_names[pred]}")

if hasattr(model, "predict_proba"):
    probs = model.predict_proba([sample])[0]
    for name, p in zip(target_names, probs):
        print(f"  {name}: {p:.1%}")
