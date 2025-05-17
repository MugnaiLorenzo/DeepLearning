import torch
import torch.nn as nn


# File: vit.py

class VisionTransformer(nn.Module):
    """
    Vision Transformer trunk: patch embedding, class token, positional embeddings and transformer blocks.
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
        # Calcola numero di patch lungo ogni dimensione
        num_patches = (img_size // patch_size) ** 2

        # 1. Proiezione lineare delle patch:
        #    da C*P*P a embed_dim
        self.patch_to_emb = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

        # 2. Token CLS: learnable e condiviso
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. Positional embeddings (includono pos per CLS + pos per patch)
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

        # 5. LayerNorm finale
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: patch tensor [B, N_patches, C, P, P]
        Returns:
            embeddings: [B, 1 + N_patches, embed_dim]
        """
        B, N, C, P, _ = x.shape
        # Flatten spatial dims and proietta
        x = x.view(B, N, C * P * P)  # [B, N, C*P*P]
        x = self.patch_to_emb(x)  # [B, N, D]

        # Espandi e concatena token CLS
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B,1,D]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]

        # Aggiungi embedding posizionali
        x = x + self.pos_emb  # [B, N+1, D]
        x = self.pos_dropout(x)

        # Passaggio nei Transformer encoder
        for blk in self.blocks:
            x = blk(x)

        # Normalizzazione finale
        x = self.norm(x)
        return x


class ViTAnchor(nn.Module):
    """
    Wrapper che mantiene due encoder: student e teacher.
    Il teacher viene aggiornato con EMA dei pesi del student.
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
            momentum: float = 0.996
    ):
        super().__init__()
        # Student network
        self.student = VisionTransformer(
            img_size, patch_size, in_channels,
            embed_dim, depth, num_heads,
            mlp_dim, dropout
        )
        # Teacher network
        self.teacher = VisionTransformer(
            img_size, patch_size, in_channels,
            embed_dim, depth, num_heads,
            mlp_dim, dropout
        )
        # Momentum per aggiornamento EMA
        self.m = momentum

        # Inizializzo teacher con pesi student e disabilito grad
        self._init_teacher()

    def _init_teacher(self):
        """Copia i pesi dallo student al teacher e disabilita grad."""
        for p_s, p_t in zip(self.student.parameters(), self.teacher.parameters()):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad_(False)

    def _update_teacher(self):
        """Aggiorna i pesi del teacher con EMA dei pesi dello student."""
        for p_s, p_t in zip(self.student.parameters(), self.teacher.parameters()):
            p_t.data = p_t.data * self.m + p_s.data * (1. - self.m)

    def forward(self, anchors: torch.Tensor, targets: torch.Tensor):
        """
        Esegue il forward dello student sulle anchor e del teacher sui target.

        Args:
            anchors: [B, M, N_patches, C, P, P]
            targets: [B, N_patches, C, P, P]

        Returns:
            feat_s: embeddings student [B, M, N+1, D]
            feat_t: embedding teacher [B, N+1, D]
        """
        B, M, N, C, P, _ = anchors.shape
        # Appiattisco M dimensione per passare nel transformer come batch
        anchors = anchors.view(B * M, N, C, P, P)
        # Student forward
        feat_s = self.student(anchors)  # [B*M, N+1, D]
        feat_s = feat_s.view(B, M, N + 1, -1)  # [B, M, N+1, D]

        # Teacher forward (no_grad)
        B_t, N_t, C_t, P_t, _ = targets.shape
        targets = targets.view(B, N, C, P, P)
        with torch.no_grad():
            feat_t = self.teacher(targets)  # [B, N+1, D]

        # EMA update
        self._update_teacher()
        return feat_s, feat_t
