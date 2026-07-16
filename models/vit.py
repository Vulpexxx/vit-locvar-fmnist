import torch
import torch.nn as nn

class LinearPatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=16, in_chans=1, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(in_chans * patch_size * patch_size, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, self.num_patches, -1)
        x = self.proj(x)
        return x


class ViT(nn.Module):
  def __init__(self, img_size=64, patch_size=16, in_chans=1, num_classes=10, 
               embed_dim=128, depth=4, num_heads=4, mlp_ratio=4.0, patch_type="conv"):
    super(ViT, self).__init__()

    self.img_size = img_size
    self.patch_size = patch_size
    self.num_patches = (img_size // patch_size) ** 2

    # 1. conv / linear Patch Embedding
    if patch_type == "conv":
        self.patch_embed = nn.Conv2d(
          in_channels=in_chans,
          out_channels=embed_dim, 
          kernel_size=patch_size,
          stride=patch_size
        )
    elif patch_type == "linear":
        self.patch_embed = LinearPatchEmbedding(img_size, patch_size, in_chans, embed_dim)
    else:
        raise ValueError("patch_type only support 'conv' or 'linear'")

   
    # 2. Class Token & Absolute Positional Encoding
   self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)

    # 3. Transformer Encoder Blocks
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=embed_dim,
      nhead=num_heads,
      dim_feedforward=int(embed_dim * mlp_ratio),
      activation='gelu',
      batch_first=True,   # Batch-first mode for input shape [batch_size, seq_len, embed_dim]
      norm_first=True     # Apply LayerNorm before attention and feedforward layers
    )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
    self.norm = nn.LayerNorm(embed_dim)

    # 4. Classification Head
    self.head = nn.Linear(embed_dim, num_classes)

  def forward(self, x):
    B = x.shape[0]
    if isinstance(self.patch_embed, nn.Conv2d):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
    else:
        x = self.patch_embed(x)
   
    # Expand and concatenate CLS token: [B, 65, embed_dim] 
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # Add positional encoding: [B, 65, embed_dim]
    x = x + self.pos_embed

    # Pass through Transformer Encoder: [B, 65, embed_dim]
    x = self.transformer(x)

    # Extract the feature corresponding to the CLS token and apply layer normalization
    cls_feat = self.norm(x[:, 0])

    # Pass through the classification head to output the final unnormalized classification scores (Logits)
    out = self.head(cls_feat)

    return out
