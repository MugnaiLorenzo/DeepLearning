import math
import os
import time
from pathlib import Path
from config import (
    batch_size, num_epochs, num_anchors, learning_rate, weight_decay,
    lambda_reg, embed_dim, depth, mlp_dim, num_heads,
    mask_ratio, random_mask_prob, patch_size, momentum,
    num_prototypes, tau, tau_plus
)
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from src.data_manager import MaskedPatchDataset, build_dataloader
from src.model.ViT import ViTAnchor
import hashlib


def tensor_hash(t):
    return hashlib.sha1(t.cpu().numpy().tobytes()).hexdigest()


def cosine_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(progress * math.pi))

    return LambdaLR(optimizer, lr_lambda)


def _iter_loader(loader, device):
    for batch in loader:
        anchors = batch['anchors'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        masks = batch['masks'].to(device, non_blocking=True)
        yield anchors, targets, masks


def train_one_epoch(model, loader, optimizer, scheduler, lambda_reg, device, scaler, epoch):
    model.train()
    total_loss, total_match, total_me = 0.0, 0.0, 0.0
    for batch_idx, (anchors, targets, masks) in enumerate(_iter_loader(loader, device), 1):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            p_s, p_t, cls_s, cls_t = model(anchors, targets)
            loss, match_loss, me_max = model.loss(p_s, p_t, lambda_reg)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        torch.cuda.empty_cache()
        total_loss += loss.item()
        total_match += match_loss.item()
        total_me += me_max.item()

        if batch_idx % 10 == 0:
            print(f"  [Batch {batch_idx}/{len(loader)}] Loss: {loss.item():.4f}")
        if batch_idx % 20 == 0:
            print(f"  🎯 Total: {loss.item():.4f} | Matching: {match_loss.item():.4f} | ME-MAX: {me_max.item():.4f}")
            print(f"🔍 [Batch {batch_idx}] Anchor[0] hash: {tensor_hash(anchors[0])}")
            print(f"🔍 [Batch {batch_idx}] Target[0] hash: {tensor_hash(targets[0])}")
            with torch.no_grad():
                top1 = p_s.argmax(dim=-1)
                bincount = top1.view(-1).bincount(minlength=model.prototypes.shape[0]).tolist()
                used = sum([1 for c in bincount if c > 0])
                print(f"    🔎 Top-1 Assignment Count: {bincount}")
                print(f"    🧠 Prototipi usati nel batch: {used}/{len(bincount)}")
    N = len(loader)
    return total_loss / N, total_match / N, total_me / N


def main():
    print("🚀 Start training")
    root = Path(__file__).parent.parent
    data_path = root / 'data' / 'imagenet-10'
    ckpt_dir = root / 'checkpoints'
    ckpt_path = ckpt_dir / "msn_best.pth"
    log_dir = "runs/msn_experiment"

    os.makedirs(ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=log_dir)
    scaler = GradScaler()

    dataset = MaskedPatchDataset(
        dataset_path=str(data_path),
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        random_mask_prob=random_mask_prob,
        num_anchors=num_anchors
    )
    loader = build_dataloader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = ViTAnchor(
        img_size=224, patch_size=patch_size, in_channels=3,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        mlp_dim=mlp_dim, dropout=0.1, momentum=momentum,
        num_prototypes=num_prototypes, tau=tau, tau_plus=tau_plus
    ).to(device)

    optimizer = optim.AdamW(
        model.student.parameters(), lr=learning_rate,
        betas=(0.9, 0.999), weight_decay=weight_decay
    )

    total_steps = num_epochs * len(loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = cosine_scheduler(optimizer, warmup_steps, total_steps)

    best_loss = float("inf")
    start_epoch = 1

    if ckpt_path.exists():
        print(f"Carico checkpoint da {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.student.load_state_dict(checkpoint['model_state'])
        model.teacher.load_state_dict(checkpoint['model_state_teacher'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Riprendo da epoca {start_epoch} con best loss {best_loss:.4f}")

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n=== Epoch {epoch}/{num_epochs} ===")
        loss, match, me = train_one_epoch(
            model, loader, optimizer, scheduler, lambda_reg, device, scaler, epoch
        )

        print(f"Epoch {epoch} completed")
        print(f"  Total Loss: {loss:.4f} | Matching Loss: {match:.4f} | ME-MAX: {me:.4f}")

        writer.add_scalar('Loss/total', loss, epoch)
        writer.add_scalar('Loss/matching', match, epoch)
        writer.add_scalar('Loss/me_max', me, epoch)

        if match < best_loss:
            best_loss = match
            torch.save({
                'epoch': epoch,
                'model_state': model.student.state_dict(),
                'model_state_teacher': model.teacher.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_loss': best_loss
            }, ckpt_path)
            print(f"✔ Salvato nuovo best model all'epoca {epoch}")

    writer.close()


if __name__ == '__main__':
    main()
