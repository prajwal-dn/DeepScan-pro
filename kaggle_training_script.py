import os
import sys

import subprocess
print("=== 0. INSTALLING DEPENDENCIES ===")
subprocess.run(["pip", "install", "-q", "av"])

print("\n=== 1. ORGANIZING DATASETS ===")
# Create the clean folder structure
os.makedirs('/kaggle/working/data/real', exist_ok=True)
os.makedirs('/kaggle/working/data/deepfake', exist_ok=True)
os.makedirs('/kaggle/working/data/ai_generated', exist_ok=True)

# Helper function to magically link the images instantly
def link_files(src, dst):
    if not os.path.exists(src): return
    for f in os.listdir(src):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            try: os.symlink(os.path.join(src, f), os.path.join(dst, f"{len(os.listdir(dst))}_{f}"))
            except: pass

# Find the datasets and organize them (BULLETPROOF METHOD)
for root, dirs, files in os.walk('/kaggle/input'):
    root_lower = root.lower()
    folder_name = os.path.basename(root).lower()
    
    if 'cifake' in root_lower:
        if folder_name == 'real': 
            link_files(root, '/kaggle/working/data/real')
        elif folder_name == 'fake': 
            link_files(root, '/kaggle/working/data/ai_generated')
            
    else:
        # For the 140k dataset (or any other dataset)
        if folder_name == 'real': 
            link_files(root, '/kaggle/working/data/real')
        elif folder_name == 'fake': 
            link_files(root, '/kaggle/working/data/deepfake')

print(f"Real Images total: {len(os.listdir('/kaggle/working/data/real'))}")
print(f"Deepfake Images total: {len(os.listdir('/kaggle/working/data/deepfake'))}")
print(f"AI Images total: {len(os.listdir('/kaggle/working/data/ai_generated'))}")
print("==============================\n")

print("=== 2. STARTING TRAINING ===")
# Paste the entirety of deepfake_detector.py here
import json, cv2, torch, timm, numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score
import tempfile

class CFG:
    DATA_ROOT         = "/kaggle/working/data"
    CKPT_DIR          = "/kaggle/working/checkpoints"
    BEST_MODEL        = "/kaggle/working/checkpoints/best_deepfake_model.pth"
    BACKBONE          = "efficientnet_b3" 
    IMG_SIZE          = 224
    EPOCHS            = 20
    BATCH_SIZE        = 32 # We can use a bigger batch size on Kaggle GPUs!
    LR                = 1e-4
    WEIGHT_DECAY      = 1e-5
    PATIENCE          = 5
    FRAMES_PER_VIDEO  = 1
    FACE_MARGIN       = 30
    DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CFG.CKPT_DIR, exist_ok=True)
print(f"[INFO] Running on: {CFG.DEVICE}")

class DeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(CFG.BACKBONE, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

class DeepfakeDataset(Dataset):
    IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(self, root, split="train", val_frac=0.15, seed=42):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        self.samples = []
        self.labels = []
        
        if split == "train":
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(CFG.IMG_SIZE, CFG.IMG_SIZE), scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.7, contrast=0.4, saturation=0.2, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.3, p=0.5),
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
                ext = fp.suffix.lower()
                if ext in self.IMAGE_EXT or ext in self.VIDEO_EXT:
                    self.samples.append((str(fp), label))
                    self.labels.append(label)
        
        print(f"[DATA] {split}: {len(self.samples)} files indexed.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        face = None
        try:
            ext = Path(path).suffix.lower()
            if ext in self.IMAGE_EXT:
                img = cv2.imread(path)
                if img is not None: 
                    face = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                import av
                container = av.open(path)
                for frame in container.decode(video=0):
                    face = frame.to_ndarray(format="rgb24")
                    break
                container.close()
        except: pass
        if face is None: face = np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE, 3), dtype=np.uint8)
        aug = self.transform(image=face)["image"]
        return aug, torch.tensor(label, dtype=torch.long)

def train_model():
    train_ds = DeepfakeDataset(CFG.DATA_ROOT, "train")
    val_ds   = DeepfakeDataset(CFG.DATA_ROOT, "val")

    targets = torch.tensor(train_ds.labels)
    num_real = (targets == 0).sum().item()
    num_df = (targets == 1).sum().item()
    num_ai = (targets == 2).sum().item()
    print(f"[SAMPLER] Balancing {num_real} Real vs {num_df} Deepfake vs {num_ai} AI Generated...")
    
    class_weights = [
        1.0 / (num_real + 1e-6), 
        1.0 / (num_df + 1e-6), 
        1.0 / (num_ai + 1e-6)
    ]
    sample_weights = torch.tensor([class_weights[t] for t in targets])
    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=2)

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
        all_labels, all_probs = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                logits = model(imgs.to(CFG.DEVICE))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.extend(probs); all_labels.extend(labels.numpy())

        try:
            val_metric = roc_auc_score(all_labels, all_probs, multi_class="ovo")
            metric_name = "AUC"
        except Exception:
            preds = np.argmax(all_probs, axis=1)
            val_metric = accuracy_score(all_labels, preds)
            metric_name = "Accuracy"

        print(f"--- Epoch {epoch} | Val {metric_name}: {val_metric:.4f} ---")
        if val_metric > best_auc:
            best_auc = val_metric
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_auc": val_metric}, CFG.BEST_MODEL)
            print("⭐ New Best Model Saved!")
        scheduler.step()

# Kick off training directly
train_model()
