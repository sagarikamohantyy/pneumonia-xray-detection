import os, copy
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_auc_score, confusion_matrix

DATA_DIR = "chest_xray"
OUT_DIR = "results"
EPOCHS = 2
BATCH = 16
LR = 1e-4
SENS_TARGET = 0.95
torch.manual_seed(0)
os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ToTensor(), norm,
])
eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), norm,
])
train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_tf)
val_ds = datasets.ImageFolder(f"{DATA_DIR}/val", transform=eval_tf)
test_ds = datasets.ImageFolder(f"{DATA_DIR}/test", transform=eval_tf)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH)
test_dl = DataLoader(test_ds, batch_size=BATCH)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 1))
for p in model.features.parameters():
    p.requires_grad = False
for p in model.features[-3:].parameters(): 
    p.requires_grad = True
model = model.to(device)

n_neg = sum(1 for _, y in train_ds.samples if y == 0)
n_pos = sum(1 for _, y in train_ds.samples if y == 1)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], device=device))
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

best_loss, best_state = float("inf"), None
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.float().to(device)
        opt.zero_grad()
        loss = criterion(model(x).squeeze(1), y)
        loss.backward()
        opt.step()
    model.eval()
    vl, n = 0.0, 0
    with torch.no_grad():
        for x, y in val_dl:
            out = model(x.to(device)).squeeze(1)
            vl += criterion(out, y.float().to(device)).item() * len(y)
            n += len(y)
    vl /= n
    print(f"Epoch {epoch+1}/{EPOCHS}  val_loss={vl:.4f}")
    if vl < best_loss:
        best_loss, best_state = vl, copy.deepcopy(model.state_dict())
model.load_state_dict(best_state)

def run_probs(dl):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in dl:
            ps.append(torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)

def metrics(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return tp / (tp + fn), tn / (tn + fp), (tp + tn) / len(y), (tn, fp, fn, tp)

val_p, val_y = run_probs(val_dl)
test_p, test_y = run_probs(test_dl)

best_t, best_spec = 0.5, -1.0
for t in np.arange(0.05, 0.996, 0.01):
    s, sp, _, _ = metrics(val_y, val_p, t)
    if s >= SENS_TARGET and sp > best_spec:
        best_t, best_spec = t, sp

s, sp, a, (tn, fp, fn, tp) = metrics(test_y, test_p, best_t)
lines = [
    f"Threshold (chosen on validation): {best_t:.2f}",
    f"Test AUC-ROC: {roc_auc_score(test_y, test_p):.3f}",
    f"Test Sensitivity: {s:.3f}",
    f"Test Specificity: {sp:.3f}",
    f"Test Accuracy: {a:.3f}",
    f"Test confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}",
]
print("\n--- Fine-tuned MobileNetV2 test results ---")
print("\n".join(lines))
with open(os.path.join(OUT_DIR, "finetune_metrics.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")