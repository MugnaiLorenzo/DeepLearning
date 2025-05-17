import torch
from torch.utils.data import DataLoader
from src.data_manager import MaskedPatchDataset, build_dataloader
from src.model.ViT import ViTAnchor
from pathlib import Path


# All’interno di main()


def main():
    root = Path(__file__).parent.parent  # Da src/vit.py a project_root
    dataset_path = root / 'data' / 'imagenet-10'
    batch_size = 1
    num_epochs = 1
    num_anchors = 1

    # Istanzia il dataset
    dataset = MaskedPatchDataset(
        dataset_path=dataset_path,
        patch_size=16,
        mask_ratio=0.5,
        random_mask_prob=0.8,
        num_anchors=num_anchors
    )

    # Costruisci il dataloader
    loader = build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )

    # Istanzia il modello ViTAnchor
    model = ViTAnchor(
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        dropout=0.1,
        momentum=0.996
    )
    model.eval()  # modalità eval per test senza dropout/training

    # Loop di test
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for i, batch in enumerate(loader):
            anchors = batch['anchors']  # [B, M, N_patches, C, P, P]
            targets = batch['targets']  # [B, N_patches, C, P, P]
            masks = batch['masks']  # [B, M, N_patches]

            # Stampa shape dei dati
            print(f" Batch {i + 1}: anchors={anchors.shape}, targets={targets.shape}, masks={masks.shape}")

            # Test forward sui due encoder
            with torch.no_grad():
                feat_s, feat_t = model(anchors, targets)

            print(f" Student output shape: {feat_s.shape}")  # [B, M, N+1, D]
            print(f" Teacher output shape: {feat_t.shape}")  # [B, N+1, D]

            # Stop after primo batch di prova
            break


if __name__ == "__main__":
    main()
