import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from src.data_manager import MaskedPatchDataset, build_dataloader
from src.model.ViT import ViTAnchor


def cosine_scheduler(optimizer, warmup_steps, total_steps):
    """
    Returns a learning rate lambda function with linear warmup and cosine decay.
    """

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, lambda_reg, device):
    model.train()
    total_loss = 0.0
    total_match = 0.0
    total_me = 0.0
    for anchors, targets, masks in _iter_loader(loader, device):
        optimizer.zero_grad()
        p_s, p_t, _, _ = model(anchors, targets)
        loss, match_loss, me_max = model.loss(p_s, p_t, lambda_reg)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        total_match += match_loss.item()
        total_me += me_max.item()
        break
    N = len(loader)
    return total_loss / N, total_match / N, total_me / N


def _iter_loader(loader, device):
    for batch in loader:
        anchors = batch['anchors'].to(device)
        targets = batch['targets'].to(device)
        yield anchors, targets, batch['masks'].to(device)


def main():
    # Configuration
    root = Path(__file__).parent.parent
    data_path = root / 'data' / 'imagenet-10'
    batch_size = 64
    num_epochs = 30
    num_anchors = 8
    learning_rate = 1e-3
    weight_decay = 0.04
    lambda_reg = 5.0

    writer = SummaryWriter(log_dir="runs/msn_experiment")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Dataset and loader
    dataset = MaskedPatchDataset(
        dataset_path=str(data_path),
        patch_size=16,
        mask_ratio=0.5,
        random_mask_prob=0.5,
        num_anchors=num_anchors
    )
    loader = build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == 'cuda'
    )

    # Model
    model = ViTAnchor(
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        dropout=0.1,
        momentum=0.996,
        num_prototypes=1024,
        tau=0.2,
        tau_plus=0.05
    ).to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.student.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=weight_decay
    )
    total_steps = num_epochs * len(loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = cosine_scheduler(optimizer, warmup_steps, total_steps)

    # Training loop
    best_loss = float('inf')
    ckpt_dir = root / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print("Start")
    for epoch in range(1, num_epochs + 1):
        loss, match, me = train_one_epoch(model, loader, optimizer, scheduler, lambda_reg, device)
        print(f"Epoch {epoch}/{num_epochs} - loss: {loss:.4f}, match: {match:.4f}, me_max: {me:.4f}")
        writer.add_scalar('Loss/total', loss, epoch)
        writer.add_scalar('Loss/matching', match, epoch)
        writer.add_scalar('Loss/me_max', me, epoch)
        # Checkpointing
        if loss < best_loss:
            best_loss = loss
            ckpt_path = ckpt_dir / f"best_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state': model.student.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_loss': best_loss
            }, ckpt_path)
    writer.close()


if __name__ == '__main__':
    main()
