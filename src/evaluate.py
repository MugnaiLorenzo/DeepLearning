import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import random
from pathlib import Path
from src.model.ViT import ViTAnchor
from src.data_manager import DataManager  # ✅ importato per patchify

# CONFIG
DATA_DIR = Path("data/imagenet-10")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
EMBED_DIM = 512
N_PER_CLASS = 50  # immagini per classe nel train set
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

print("📁 Carico dataset da:", DATA_DIR)

# Trasformazioni coerenti con il training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.convert("RGB")),  # for safety
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255.0),  # riportiamo al range [0, 255]
])

# Dataset completo
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
print(f"📷 Dataset totale: {len(full_dataset)} immagini in {NUM_CLASSES} classi")

# Split bilanciato train/test
class_indices = {i: [] for i in range(NUM_CLASSES)}
for idx, (_, label) in enumerate(full_dataset):
    class_indices[label].append(idx)

train_indices, test_indices = [], []
for cls, indices in class_indices.items():
    random.shuffle(indices)
    selected_train = indices[:N_PER_CLASS]
    selected_test = indices[N_PER_CLASS:]
    train_indices.extend(selected_train)
    test_indices.extend(selected_test)
    print(f"🔠 Classe {cls}: {len(selected_train)} train, {len(selected_test)} test")

train_set = Subset(full_dataset, train_indices)
test_set = Subset(full_dataset, test_indices)

train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# Carica modello MSN e congela tutto
print("📦 Carico modello pre-addestrato MSN...")
model = ViTAnchor(
    img_size=224, patch_size=16, in_channels=3,
    embed_dim=EMBED_DIM, depth=8, num_heads=8,
    mlp_dim=2048, dropout=0.1, momentum=0.996,
    num_prototypes=1024, tau=0.07, tau_plus=0.07
).to(DEVICE)

ckpt = torch.load("checkpoints/msn_best.pth", map_location=DEVICE)
model.student.load_state_dict(ckpt['model_state'])
model.teacher.load_state_dict(ckpt['model_state_teacher'])  # deve esistere nel checkpoint
model.eval()

for p in model.teacher.parameters():
    p.requires_grad = False

# 🔧 Istanza patcher per immagini
patcher = DataManager(patch_size=16)

# Funzione per estrarre embedding CLS
@torch.no_grad()
def extract_features(dataloader, tag="train"):
    features, labels = [], []
    print(f"🔄 Estrazione feature da {tag} set...")
    for i, (x, y) in enumerate(dataloader):
        x = x.to(DEVICE)  # [B, 3, 224, 224]
        x_patch = torch.stack([patcher.patchify(img) for img in x])  # [B, N, C, P, P]
        z = model.teacher(x_patch)[:, 0, :]  # CLS token [B, D]
        features.append(z.cpu().numpy())
        labels.extend(y.numpy())
        if i % 5 == 0:
            print(f"  > Batch {i + 1} processed")
    return np.concatenate(features), np.array(labels)

print("🔍 Inizio estrazione feature...")
X_train, y_train = extract_features(train_loader, tag='train')
X_test, y_test = extract_features(test_loader, tag='test')

# Analisi diagnostica (debug facoltativo)
print(f"🔬 Media feature train (primi 5): {np.mean(X_train, axis=0)[:5]}")
print(f"🔬 Std feature train (primi 5): {np.std(X_train, axis=0)[:5]}")

print("🧠 Addestramento Logistic Regression su embeddings...")
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)
print("✅ Classificatore addestrato.")

print("📊 Valutazione sulle immagini di test...")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy con {N_PER_CLASS} esempi per classe: {acc * 100:.2f}%")
