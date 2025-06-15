import os
import random
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, datasets
from PIL import Image
from torch.utils.data import Subset

class DataManager:
    """
    Utility class per gestione di trasformazioni e operazioni su singola immagine.
    """

    def __init__(self, patch_size: int = 16,
                 mask_ratio: float = 0.5, random_mask_prob: float = 0.8):
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.random_mask_prob = random_mask_prob

    def get_augmentations(self):
        """Augmentazioni per target e anchor views."""
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomResizedCrop(224, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(3),
            T.ToTensor(),
        ])

    def patchify(self, img_tensor: torch.Tensor):
        """Suddivide l'immagine in patch quadrate."""
        C, H, W = img_tensor.shape
        num_h = H // self.patch_size
        num_w = W // self.patch_size
        patches = []
        for i in range(num_h):
            for j in range(num_w):
                h0 = i * self.patch_size
                w0 = j * self.patch_size
                patches.append(
                    img_tensor[:, h0:h0 + self.patch_size, w0:w0 + self.patch_size]
                )
        return torch.stack(patches, dim=0)

    def random_mask(self, patches: torch.Tensor):
        """Applica masking casuale alle patch."""
        total = patches.size(0)
        num_mask = int(self.mask_ratio * total)
        idxs = random.sample(range(total), num_mask)
        mask = torch.zeros(total, dtype=torch.bool)
        mask[idxs] = True
        masked = patches.clone()
        masked[mask] = 0
        return masked, mask

    def focal_mask(self, patches: torch.Tensor, grid_size: int = 4):
        """Masking focalizzato: blocco contiguo di patch grid_size×grid_size mascherato."""
        num_patches = patches.size(0)
        side = int(num_patches ** 0.5)
        assert side * side == num_patches, "Numero di patch non quadrato"
        # Crea una matrice di mask inizialmente tutta True (nessuna patch mascherata)
        mask_matrix = torch.ones(side, side, dtype=torch.bool)
        # Seleziona casualmente il blocco da mascherare
        x = random.randint(0, side - grid_size)
        y = random.randint(0, side - grid_size)
        # Imposta a False le patch mascherate
        mask_matrix[x:x + grid_size, y:y + grid_size] = False
        # Appiattisci in un vettore booleano
        mask = ~mask_matrix.view(-1)
        # Applica la maschera alle patch
        masked = patches.clone()
        masked[mask] = 0
        return masked, mask

    def choose_mask_type(self):
        """Seleziona tipo di masking in base a probabilità.""(self):"""
        return 'random' if random.random() < self.random_mask_prob else 'focal'


class MaskedPatchDataset(DataManager, Dataset):
    """Dataset che restituisce M viste mascherate, patch target e mask per ogni campione."""

    def __init__(
            self,
            dataset_path: str,
            patch_size: int = 16,
            mask_ratio: float = 0.5,
            random_mask_prob: float = 0.8,
            num_anchors: int = 2,
    ):
        DataManager.__init__(self, patch_size, mask_ratio, random_mask_prob)
        self.dataset = datasets.ImageFolder(root=dataset_path)
        self.samples = self.dataset.samples
        self.num_anchors = num_anchors

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, _ = self.samples[idx]
        img = Image.open(path).convert('RGB')

        # Augmentazione per target
        aug = self.get_augmentations()
        target_view = aug(img)
        patches_target = self.patchify(target_view)

        # Generazione di M viste anchor mascherate
        anchor_views = []
        masks = []
        for _ in range(self.num_anchors):
            view = aug(img)
            patches_anchor = self.patchify(view)
            mask_type = self.choose_mask_type()
            if mask_type == 'random':
                masked, mask = self.random_mask(patches_anchor)
            else:
                masked, mask = self.focal_mask(patches_anchor)
            anchor_views.append(masked)
            masks.append(mask)

        # Stack delle viste e maschere
        anchors = torch.stack(anchor_views, dim=0)  # [M, N_patches, C, P, P]
        masks = torch.stack(masks, dim=0)  # [M, N_patches]

        return {
            'anchors': anchors,
            'targets': patches_target,
            'masks': masks,
        }


def build_dataloader(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
) -> DataLoader:
    """Costruisce il DataLoader per il MaskedPatchDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
