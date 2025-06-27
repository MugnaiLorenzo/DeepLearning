
import torch
import numpy as np
from src.model.ViT import ViTAnchor
from src.data_manager import DataManager
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "app" / "msn_best1.pth"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "imagenet-10"

# Costruzione modello
model = ViTAnchor(
    img_size=224, patch_size=16, in_channels=3,
    embed_dim=512, depth=8, num_heads=8,
    mlp_dim=2048, dropout=0.1, momentum=0.996,
    num_prototypes=1024, tau=0.07, tau_plus=0.07
).to(DEVICE)

# Caricamento checkpoint
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
print("📂 Checkpoint caricato da epoca:", ckpt.get("epoch", "Non specificata"))

model.student.load_state_dict(ckpt['model_state'])
model.teacher.load_state_dict(ckpt['model_state_teacher'])

# Statistiche CLS token
cls_token = model.teacher.cls_token.view(-1).detach().cpu().numpy()
print("🧪 Teacher CLS token:")
print("    Min:", np.min(cls_token))
print("    Max:", np.max(cls_token))
print("    Std:", np.std(cls_token))

# Statistiche prototipi
prototypes = model.prototypes.detach().cpu().numpy()
proto_mean = np.mean(prototypes, axis=0)
proto_std = np.std(prototypes, axis=0)
print("🔎 Prototipi:")
print("    Media media:", np.mean(proto_mean))
print("    Deviazione std media:", np.mean(proto_std))

# Verifica embedding reali su immagini
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255.0),
])
dataset = datasets.ImageFolder(root=str(DATA_DIR), transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=False)

patcher = DataManager(patch_size=16)
model.eval()

with torch.no_grad():
    for x, _ in loader:
        x = x.to(DEVICE)
        x_patch = torch.stack([patcher.patchify(img) for img in x])
        z = model.teacher(x_patch)[:, 0, :]  # CLS
        z = z.cpu().numpy()
        break

print("📌 Primo embedding CLS (primi 10 valori):", z[0][:10])
print("📏 Norma differenza tra primi due embedding:", np.linalg.norm(z[0] - z[1]))
print("📈 Std media tra tutti i 4 embedding:", np.mean(np.std(z, axis=0)))
print("🎨 Differenza tra x_patch[0] e x_patch[1]:", torch.norm(x_patch[0] - x_patch[1]).item())
print("🧪 Media embedding input x_patch[0]:", x_patch[0].mean().item())
print("🧪 Media embedding output z[0]:", z[0].mean())
