import os, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "chest_xray"
OUT_DIR = "results"
EPOCHS = 8
BATCH = 32
SENS_TARGET = 0.95
torch.manual_seed(0)
os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=tf)
val_ds = datasets.ImageFolder(f"{DATA_DIR}/val", transform=tf)
test_ds = datasets.ImageFolder(f"{DATA_DIR}/test", transform=tf)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH)
test_dl = DataLoader(test_ds, batch_size=BATCH)

def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                         nn.ReLU(), nn.MaxPool2d(2))

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            block(1, 32), block(32, 64), block(64, 128),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        a = self.features(x)
        return self.head(self.drop(self.pool(a).flatten(1))).squeeze(1)

model = SmallCNN().to(device)
n_neg = sum(1 for _, y in train_ds.samples if y == 0)
n_pos = sum(1 for _, y in train_ds.samples if y == 1)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], device=device))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

best_loss, best_state = float("inf"), None
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.float().to(device)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    vl, n = 0.0, 0
    with torch.no_grad():
        for x, y in val_dl:
            vl += criterion(model(x.to(device)), y.float().to(device)).item() * len(y)
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
            ps.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
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
print("\n--- CNN test results ---")
print("\n".join(lines))
with open(os.path.join(OUT_DIR, "cnn_metrics.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
torch.save(best_state, os.path.join(OUT_DIR, "cnn_weights.pt"))

def gradcam(x):
    model.eval()
    x = x.unsqueeze(0).to(device)
    a = model.features(x)
    a.retain_grad()
    logit = model.head(model.pool(a).flatten(1))
    model.zero_grad()
    logit.sum().backward()
    w = a.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(128, 128), mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return cam / (cam.max() + 1e-8)

pred = test_p >= best_t
tp_idx = np.where((test_y == 1) & pred)[0][:2]
fp_idx = np.where((test_y == 0) & pred)[0][:2]
idxs = list(tp_idx) + list(fp_idx)
if idxs:
    fig, axes = plt.subplots(2, len(idxs), figsize=(3 * len(idxs), 6))
    for col, i in enumerate(idxs):
        x, y = test_ds[i]
        img = (x[0] * 0.5 + 0.5).numpy()
        cam = gradcam(x)
        label = "Pneumonia" if y == 1 else "Normal"
        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(f"{label}, p={test_p[i]:.2f}", fontsize=9)
        axes[1, col].imshow(img, cmap="gray")
        axes[1, col].imshow(cam, cmap="jet", alpha=0.4)
        axes[1, col].set_title("Grad-CAM", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "gradcam_examples.png"), dpi=150)
print(f"\nSaved results to {OUT_DIR}")