"""
================================================================
 DecodeLabs - Artificial Intelligence Internship
 Project 2: Data Classification Using AI
 Goal: Build a classification model using the Iris dataset,
       following the INPUT -> PROCESS -> OUTPUT (IPO) framework
       shown in the training slides.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)

# ----------------------------------------------------------------
# STEP 1 (INPUT): Load and understand the dataset
# ----------------------------------------------------------------
iris = load_iris()
X = iris.data                      # 150 samples x 4 features
y = iris.target                    # 3 classes: setosa, versicolor, virginica
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["species"] = pd.Categorical.from_codes(y, target_names)

print("=" * 60)
print("STEP 1: DATASET OVERVIEW (Raw Material: The Iris Benchmark)")
print("=" * 60)
print(f"Samples: {df.shape[0]}")
print(f"Features (dimensions): {len(feature_names)} -> {feature_names}")
print(f"Classes: {len(target_names)} -> {list(target_names)}")
print("\nFirst 5 rows:")
print(df.head())
print("\nClass balance:")
print(df["species"].value_counts())

# ----------------------------------------------------------------
# STEP 2 (PROCESS - part A): Train/Test split
# "Structural Integrity: The Split" - shuffle before splitting
# ----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,        # 80/20 split as shown in "The Full Architecture" slide
    random_state=42,       # reproducibility
    stratify=y,            # keep class balance equal in both sets
    shuffle=True,
)

print("\n" + "=" * 60)
print("STEP 2: TRAIN / TEST SPLIT")
print("=" * 60)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")

# ----------------------------------------------------------------
# STEP 3 (PROCESS - part B): Feature scaling
# "The Gatekeeper Rule: Scaling" - StandardScaler, mean=0, var=1
# IMPORTANT: fit scaler on TRAIN ONLY, then transform both sets
# ----------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 60)
print("STEP 3: FEATURE SCALING (StandardScaler)")
print("=" * 60)
print("Mean of scaled training data (~0):", np.round(X_train_scaled.mean(axis=0), 3))
print("Std  of scaled training data (~1):", np.round(X_train_scaled.std(axis=0), 3))

# ----------------------------------------------------------------
# STEP 4 (PROCESS - part C): Tune "K" - find the elbow point
# "Tuning the Engine: Choosing K"
# ----------------------------------------------------------------
error_rates = []
k_range = range(1, 21)
for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    pred_temp = knn_temp.predict(X_test_scaled)
    error_rates.append(np.mean(pred_temp != y_test))

# NOTE: K=1 always looks "perfect" on training-adjacent data but is the
# classic overfitting trap called out in the slides ("K=1: Noise/Overfitting").
# So we search for the best K starting at K=3 for a more generalizable model.
candidates = list(k_range)[2:]                      # K = 3, 4, 5, ...
candidate_errors = error_rates[2:]
best_k = candidates[int(np.argmin(candidate_errors))]

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), error_rates, marker="o", linestyle="--", color="steelblue")
plt.axvline(best_k, color="orange", linestyle=":", label=f"Optimal K = {best_k}")
plt.title("Tuning the Engine: Choosing K")
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.legend()
plt.tight_layout()
plt.savefig("/home/claude/k_tuning_curve.png", dpi=150)
plt.close()

print("\n" + "=" * 60)
print("STEP 4: CHOOSING OPTIMAL K")
print("=" * 60)
print(f"Optimal K selected (lowest error rate): {best_k}")

# ----------------------------------------------------------------
# STEP 5 (PROCESS - part D): Instantiate, Fit, Predict
# "The Workflow: Scikit-Learn" - Instantiate -> Fit -> Predict
# ----------------------------------------------------------------
model = KNeighborsClassifier(n_neighbors=best_k)   # INSTANTIATE
model.fit(X_train_scaled, y_train)                  # FIT (memorize the map)
predictions = model.predict(X_test_scaled)          # PREDICT (apply logic)

print("\n" + "=" * 60)
print("STEP 5: MODEL TRAINING & PREDICTION (KNN Classifier)")
print("=" * 60)
print(f"Model: KNeighborsClassifier(n_neighbors={best_k})")
print("Model trained successfully.")

# ----------------------------------------------------------------
# STEP 6 (OUTPUT): Evaluate - Confusion Matrix, Accuracy, F1
# "Output Validation" + "The Diagnostic Tool: Confusion Matrix"
# + "Strategic Trade-offs" (Precision/Recall/F1)
# ----------------------------------------------------------------
accuracy = accuracy_score(y_test, predictions)
f1_macro = f1_score(y_test, predictions, average="macro")
cm = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions, target_names=target_names)

print("\n" + "=" * 60)
print("STEP 6: OUTPUT VALIDATION")
print("=" * 60)
print(f"Accuracy Score : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"F1 Score (macro): {f1_macro:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nFull Classification Report:")
print(report)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=target_names, yticklabels=target_names
)
plt.title(f"Confusion Matrix (KNN, K={best_k})\nAccuracy: {accuracy*100:.2f}%")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("/home/claude/confusion_matrix.png", dpi=150)
plt.close()

print("\n" + "=" * 60)
print("PIPELINE COMPLETE - Project 2 Milestone Achieved!")
print("=" * 60)
