"""
KAGGLE TRAINING SCRIPT — DEEPSCAN PRO (FULL VERSION)
===================================================
Run this on Kaggle using GPU T4 x2 or P100.
Make sure to add these datasets:
1. CIFAKE: Real vs AI-Generated Images (by birdy654)
2. 140k Real and Fake Faces (by xhlulu)
"""

import os, sys, json, cv2, torch, timm, numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score
import tempfile
import subprocess

# ── 1. INSTALL DEPENDENCIES ───────────────────────────────────
print("=== 1. INSTALLING DEPENDENCIES ===")
subprocess.run(["pip", "install", "-q", "av", "timm", "albumentations"])

# ── 2. CONFIG ─────────────────────────────────────────────────
class CFG:
    DATA_ROOT         = "/kaggle/working/data"
    CKPT_DIR          = "/kaggle/working/checkpoints"
    BEST_MODEL        = "/kaggle/working/checkpoints/best_deepfake_model.pth"

    BACKBONE          = "efficientnet_b4" 
    IMG_SIZE          = 299 # High-res for forensic detail
    EPOCHS            = 5  # 5 epochs on 280k images ≈ 10 hours (safe for Kaggle)
    BATCH_SIZE        = 16 # Adjust to 8 if you get Out of Memory
    LR                = 8e-5
    WEIGHT_DECAY      = 1e-5
    PATIENCE          = 5
    DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CFG.CKPT_DIR, exist_ok=True)

# ── 3. DATA ORGANIZATION (NUCLEAR SEARCH MODE) ────────────────
print("\n=== 2. SCANNING ALL INPUT FOLDERS ===")
os.makedirs('/kaggle/working/data/real', exist_ok=True)
os.makedirs('/kaggle/working/data/deepfake', exist_ok=True)
os.makedirs('/kaggle/working/data/ai_generated', exist_ok=True)

def link_files(src, dst, limit=10000):  # 10k per folder = ~140k total (faster training)
    files = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not files: return
    print(f"-> Linking {min(len(files), limit)} images from: ...{src[-40:]}")
    for f in files[:limit]:
        try:
            target = os.path.join(dst, f"{len(os.listdir(dst))}_{f}")
            if not os.path.exists(target): os.symlink(os.path.join(src, f), target)
        except: pass

def folder_label(root):
    """Classify by the actual leaf folder, not the dataset slug."""
    leaf = Path(root).name.lower().replace(" ", "_").replace("-", "_")
    parent = Path(root).parent.name.lower().replace(" ", "_").replace("-", "_")

    if leaf in {"real", "real_images", "authentic", "original"}:
        return "real"
    if leaf in {"fake", "fakes", "deepfake", "deepfakes", "face_swap", "faceswap"}:
        if "cifake" in str(Path(root)).lower() or "ai" in parent:
            return "ai_generated"
        return "deepfake"
    if leaf in {"ai", "ai_generated", "generated", "synthetic"}:
        return "ai_generated"
    return None

# Recursively crawl EVERYTHING in /kaggle/input
for root, dirs, files in os.walk('/kaggle/input'):
    label = folder_label(root)
    if label == "real":
        link_files(root, '/kaggle/working/data/real')
    elif label == "ai_generated":
        link_files(root, '/kaggle/working/data/ai_generated')
    elif label == "deepfake":
        link_files(root, '/kaggle/working/data/deepfake')

print(f"\n✅ Total Real: {len(os.listdir('/kaggle/working/data/real'))}")
print(f"✅ Total Deepfake: {len(os.listdir('/kaggle/working/data/deepfake'))}")
print(f"✅ Total AI Generated: {len(os.listdir('/kaggle/working/data/ai_generated'))}")

# ── 4. MODEL ──────────────────────────────────────────────────
class DeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(CFG.BACKBONE, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 3),
        )
    def forward(self, x): return self.head(self.backbone(x))

# ── 5. DATASET ────────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    def __init__(self, root, split="train", val_frac=0.15, seed=42):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        self.samples, self.labels = [], []
        
        if split == "train":
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(CFG.IMG_SIZE, CFG.IMG_SIZE), scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, p=0.5),
                A.GaussNoise(p=0.2),
                A.ImageCompression(quality_lower=60, p=0.3),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2(),
            ])
            
        rng = np.random.default_rng(seed)
        for label, folder in [(0, "real"), (1, "deepfake"), (2, "ai_generated")]:
            folder_path = Path(root) / folder
            if not folder_path.exists(): continue
            files = list(folder_path.iterdir())
            rng.shuffle(files)
            n_val = int(len(files) * val_frac)
            files = files[n_val:] if split == "train" else files[:n_val]
            for fp in files:
                self.samples.append(str(fp))
                self.labels.append(label)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx], self.labels[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.transform(image=img)["image"]
        return aug, torch.tensor(label, dtype=torch.long)

# ── 6. TRAINING LOOP ──────────────────────────────────────────
def train_model():
    train_ds = DeepfakeDataset(CFG.DATA_ROOT, "train")
    val_ds   = DeepfakeDataset(CFG.DATA_ROOT, "val")

    targets = torch.tensor(train_ds.labels)
    label_counts = np.bincount(np.array(train_ds.labels), minlength=3)
    print(f"[SAMPLER] Balancing Real={label_counts[0]}, Deepfake={label_counts[1]}, AI Generated={label_counts[2]}")
    class_weights = [1.0 / (label_counts[i] + 1e-6) for i in range(3)]
    sample_weights = torch.tensor([class_weights[t] for t in targets])
    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, sampler=sampler, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False)

    model = DeepfakeDetector(pretrained=True).to(CFG.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)

    best_auc = 0.0
    for epoch in range(1, CFG.EPOCHS + 1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG.EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(CFG.DEVICE), labels.to(CFG.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{train_loss/len(train_loader):.4f}")

        model.eval()
        all_labels, all_probs, all_preds = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                logits = model(imgs.to(CFG.DEVICE))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_probs.extend(probs); all_labels.extend(labels.numpy()); all_preds.extend(preds)

        # Crash-proof evaluation
        try:
            val_metric = roc_auc_score(all_labels, all_probs, multi_class="ovo")
            metric_name = "AUC"
        except:
            val_metric = accuracy_score(all_labels, all_preds)
            metric_name = "ACC"
            
        print(f"--- Epoch {epoch} | Val {metric_name}: {val_metric:.4f} ---")
        
        if val_metric > best_auc:
            best_auc = val_metric
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_metric": val_metric}, CFG.BEST_MODEL)
            print(f"⭐ New Best Model Saved (Val {metric_name}: {val_metric:.4f})")
        scheduler.step()

if __name__ == "__main__":
    train_model()

# ── 7. AUTO-SAVE TO OUTPUT ROOT ───────────────────────────────
import shutil
print(f"\n🏆 TRAINING COMPLETE!")

# Copy to root of /kaggle/working so it shows up in Output tab
dst = '/kaggle/working/best_deepfake_model.pth'
if os.path.exists(CFG.BEST_MODEL):
    shutil.copy(CFG.BEST_MODEL, dst)
    size_mb = os.path.getsize(dst) / 1024 / 1024
    print(f"✅ Model copied to Output tab: {dst}")
    print(f"✅ File size: {size_mb:.1f} MB")
    print("👉 Go to the OUTPUT tab on the right to download it!")
else:
    print("❌ ERROR: Model file not found! Training may have failed.")
