import torch, torch.nn as nn, numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

DATA_DIR = "chest_xray"   # run this script from the project root folder

tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Pretrained CNN used only as a fixed feature extractor
backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
backbone.classifier = nn.Identity()   # outputs a 1280-number feature vector
backbone.eval()

def extract(folder):
    ds = datasets.ImageFolder(folder, transform=tf)
    dl = DataLoader(ds, batch_size=32, num_workers=0)
    feats, labels = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(dl):
            feats.append(backbone(x).numpy())
            labels.append(y.numpy())
            if i % 20 == 0:
                print(f"  batch {i}/{len(dl)}")
    return np.vstack(feats), np.concatenate(labels)

print("Extracting training features...")
Xtr, ytr = extract(f"{DATA_DIR}/train")
print("Extracting test features...")
Xte, yte = extract(f"{DATA_DIR}/test")

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(Xtr, ytr)

probs = clf.predict_proba(Xte)[:, 1]
preds = clf.predict(Xte)
tn, fp, fn, tp = confusion_matrix(yte, preds).ravel()

print("\n--- Test set results ---")
print("AUC-ROC:    ", round(roc_auc_score(yte, probs), 4))
print("Sensitivity:", round(tp / (tp + fn), 4))
print("Specificity:", round(tn / (tn + fp), 4))
print("Accuracy:   ", round((tp + tn) / len(yte), 4))