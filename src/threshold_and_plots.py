import os
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "chest_xray"
OUT_DIR = "results"
CACHE = "features_cache.npz"
SENS_TARGET = 0.95
os.makedirs(OUT_DIR, exist_ok=True)

tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
backbone.classifier = nn.Identity()
backbone.eval()

def extract(folder):
    ds = datasets.ImageFolder(folder, transform=tf)
    dl = DataLoader(ds, batch_size=32, num_workers=0)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in dl:
            feats.append(backbone(x).numpy())
            labels.append(y.numpy())
    return np.vstack(feats), np.concatenate(labels)

if os.path.exists(CACHE):
    d = np.load(CACHE)
    Xtr, ytr, Xva, yva, Xte, yte = (d[k] for k in ["Xtr", "ytr", "Xva", "yva", "Xte", "yte"])
else:
    print("Extracting features (first run only)...")
    Xtr, ytr = extract(f"{DATA_DIR}/train")
    Xva, yva = extract(f"{DATA_DIR}/val")
    Xte, yte = extract(f"{DATA_DIR}/test")
    np.savez(CACHE, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(Xtr, ytr)

def get_probs(X):
    return clf.predict_proba(X)[:, 1]

def metrics(y, probs, t):
    preds = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / len(y)
    return sens, spec, acc, (tn, fp, fn, tp)

val_probs = get_probs(Xva)
grid = np.arange(0.05, 0.996, 0.01)
candidates = []
for t in grid:
    s, p, a, _ = metrics(yva, val_probs, t)
    if s >= SENS_TARGET:
        candidates.append((t, p))
if candidates:
    best_t = max(candidates, key=lambda c: c[1])[0]
else:
    best_t = 0.5
    print(f"No threshold reached sensitivity {SENS_TARGET} on validation; using 0.50")

print(f"Chosen threshold (from validation): {best_t:.2f}")
vs, vp, va, _ = metrics(yva, val_probs, best_t)
print(f"Validation: sensitivity {vs:.3f}, specificity {vp:.3f}, accuracy {va:.3f}")

test_probs = get_probs(Xte)
s, p, a, (tn, fp, fn, tp) = metrics(yte, test_probs, best_t)
print(f"\n--- Test set results at threshold {best_t:.2f} ---")
print(f"AUC-ROC:     {roc_auc_score(yte, test_probs):.3f}")
print(f"Sensitivity: {s:.3f}")
print(f"Specificity: {p:.3f}")
print(f"Accuracy:    {a:.3f}")
print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"Test counts: {tn+fp} normal, {fn+tp} pneumonia")

fpr, tpr, _ = roc_curve(yte, test_probs)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(yte, test_probs):.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate (sensitivity)")
plt.title("ROC curve, test set")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=150, bbox_inches="tight")

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, t in zip(axes, [0.5, best_t]):
    cm = confusion_matrix(yte, (test_probs >= t).astype(int), labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xticks([0, 1], ["Normal", "Pneumonia"])
    ax.set_yticks([0, 1], ["Normal", "Pneumonia"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Threshold = {t:.2f}")
plt.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved plots to {OUT_DIR}")