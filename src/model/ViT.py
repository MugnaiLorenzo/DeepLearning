import torch
import torch.nn as nn

# File: vit.py

class VisionTransformer(nn.Module):
    """
    Vision Transformer trunk:
    - Patch embedding
    - Class token
    - Positional embeddings
    - Transformer encoder blocks
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        # Numero di patch per immagine
        num_patches = (img_size // patch_size) ** 2

        # 1. Patch embedding: da C*P*P a embed_dim
        self.patch_to_emb = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

        # 2. Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. Learnable positional embeddings per CLS + tutte le patch
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # 4. Transformer encoder blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation='gelu'
            )
            for _ in range(depth)
        ])

        # 5. LayerNorm finale sull'embedding
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor di patch [B, N_patches, C, P, P]
        Returns:
            Tensor di embedding [B, N_patches+1, embed_dim]
        """
        B, N, C, P, _ = x.shape
        # Flatten delle patch
        x = x.view(B, N, C * P * P)              # [B, N, C*P*P]
        # Proiezione in embedding
        x = self.patch_to_emb(x)                # [B, N, D]

        # Concatenazione del token CLS espanso
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B,1,D]
        x = torch.cat([cls_tokens, x], dim=1)          # [B, N+1, D]

        # Aggiunta dei positional embeddings
        x = x + self.pos_emb                           # [B, N+1, D]
        x = self.pos_dropout(x)

        # Passaggio nei blocchi Transformer
        for blk in self.blocks:
            x = blk(x)

        # Normalizzazione finale
        x = self.norm(x)                              # [B, N+1, D]
        return x


class ViTAnchor(nn.Module):
    """
    Masked Siamese network wrapper:
    - Student and teacher VisionTransformer
    - EMA update for teacher
    - Prototype head
    - Loss computation (matching + ME-MAX)
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        momentum: float = 0.996,
        num_prototypes: int = 1024,
        tau: float = 0.1,
        tau_plus: float = 0.025
    ):
        super().__init__()
        # Student and teacher networks
        self.student = VisionTransformer(
            img_size, patch_size, in_channels,
            embed_dim, depth, num_heads,
            mlp_dim, dropout
        )
        self.teacher = VisionTransformer(
            img_size, patch_size, in_channels,
            embed_dim, depth, num_heads,
            mlp_dim, dropout
        )
        # EMA momentum
        self.m = momentum

        # Prototype head: K learnable prototypes of dimension D
        self.num_prototypes = num_prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim))

        # Temperatures
        self.tau = tau
        self.tau_plus = tau_plus

        # Initialize teacher parameters
        self._init_teacher()

    def _init_teacher(self):
        # Copy student weights to teacher and freeze teacher
        for p_s, p_t in zip(self.student.parameters(), self.teacher.parameters()):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad_(False)

    def _update_teacher(self):
        # EMA update: teacher = m*teacher + (1-m)*student
        for p_s, p_t in zip(self.student.parameters(), self.teacher.parameters()):
            p_t.data.mul_(self.m).add_(p_s.data, alpha=1 - self.m)

    def forward(self, anchors: torch.Tensor, targets: torch.Tensor):
        """
        Forward pass:
        - student processes masked anchor views
        - teacher processes full target views (no grad)
        - compute logits over prototypes
        """
        B, M, N, C, P, _ = anchors.shape
        # Student branch: flatten anchor batch
        anchors_flat = anchors.view(B * M, N, C, P, P)
        feat_s = self.student(anchors_flat)        # [B*M, N+1, D]
        feat_s = feat_s.view(B, M, N+1, -1)        # [B, M, N+1, D]
        cls_s = feat_s[:, :, 0, :]                # [B, M, D]

        # Teacher branch
        with torch.no_grad():
            feat_t = self.teacher(targets)        # [B, N+1, D]
            cls_t = feat_t[:, 0, :]               # [B, D]

        # EMA update of teacher
        self._update_teacher()

        # Prototype matching
        # Student logits: [B, M, K]
        logits_s = torch.einsum('bmd,kd->bmk', cls_s, self.prototypes) / self.tau
        # Teacher logits: [B, K]
        logits_t = torch.einsum('bd,kd->bk', cls_t, self.prototypes) / self.tau_plus

        # Softmax distributions
        p_s = torch.softmax(logits_s, dim=-1)    # [B, M, K]
        p_t = torch.softmax(logits_t, dim=-1)    # [B, K]
        return p_s, p_t, cls_s, cls_t

    def loss(self, p_s: torch.Tensor, p_t: torch.Tensor, lambda_reg: float):
        """
        Compute matching loss and ME-MAX regularization.

        Args:
            p_s: student distribution [B, M, K]
            p_t: teacher distribution [B, K]
            lambda_reg: weight for ME-MAX term

        Returns:
            loss: total loss
            match_loss: cross-entropy component
            me_max: entropy regularization component
        """
        B, M, K = p_s.shape
        # Expand teacher distribution to student shape
        p_t_exp = p_t.unsqueeze(1).expand_as(p_s)  # [B, M, K]
        # Matching loss: -sum(p_t * log p_s) / (B*M)
        match_loss = -(p_t_exp * torch.log(p_s + 1e-6)).sum() / (B * M)
        # ME-MAX: entropy of mean prediction
        p_bar = p_s.mean(dim=(0, 1))                # [K]
        entropy = -(p_bar * torch.log(p_bar + 1e-6)).sum()
        me_max = -lambda_reg * entropy
        loss = match_loss + me_max
        return loss, match_loss, me_max
