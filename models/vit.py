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


class TransformerBlock(nn.Module):
  """
  Custom Transformer block replacing nn.TransformerEncoderLayer.
  Includes Pre-LN multi-head attention and feed-forward network.
  Designed to output attention weights.
  """
  def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
    super().__init__()
    # Pre-LN structure (equivalent to norm_first=True)
    self.norm1 = nn.LayerNorm(embed_dim)
    self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    
    self.norm2 = nn.LayerNorm(embed_dim)
    self.ffn = nn.Sequential(
      nn.Linear(embed_dim, ff_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(ff_dim, embed_dim),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    # 1. Multi-Head Attention (Pre-LN)
    residual = x
    x = self.norm1(x)
    
    # Set need_weights=True and average_attn_weights=False to retrieve independent head weights
    attn_out, attn_weight = self.attn(x, x, x, need_weights=True, average_attn_weights=False)   
    x = residual + attn_out

    # 2. Feed Forward Network (Pre-LN)
    residual = x
    x = self.norm2(x)
    x = residual + self.ffn(x)
    
    return x, attn_weight


class ViT(nn.Module):
  def __init__(self, img_size=64, patch_size=8, in_chans=1, num_classes=10, 
               embed_dim=128, depth=4, num_heads=4, mlp_ratio=4.0, patch_type="conv"):
    super(ViT, self).__init__()

    self.img_size = img_size
    self.patch_size = patch_size
    self.num_patches = (img_size // patch_size) ** 2

    # 1. Patch Embedding
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
      raise ValueError("patch_type only supports 'conv' or 'linear'")

    # 2. Class Token & Absolute Positional Encoding
    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)

    # 3. Custom Transformer Encoder Blocks
    self.blocks = nn.ModuleList([
      TransformerBlock(embed_dim, num_heads, int(embed_dim * mlp_ratio))
      for _ in range(depth)
    ])
    self.norm = nn.LayerNorm(embed_dim)

    # 4. Classification Head
    self.head = nn.Linear(embed_dim, num_classes)

  def forward(self, x):
    """
    Default forward pass.
    Returns only classification logits to maintain compatibility with the training script.
    """
    out, _ = self.forward_with_attn(x)
    return out

  def forward_with_attn(self, x):
    """
    Forward pass that records attention weights.
    Returns a tuple containing the classification logits and the list of attention weights.
    """
    B = x.shape[0]
    
    # Extract patch features
    if isinstance(self.patch_embed, nn.Conv2d):
      x = self.patch_embed(x).flatten(2).transpose(1, 2)
    else:
      x = self.patch_embed(x)
    
    # Expand and concatenate CLS token
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # Add positional encoding
    x = x + self.pos_embed

    # Pass through Transformer blocks and record attention weights
    attn_weights = []
    for block in self.blocks:
      x, attn = block(x) 
      attn_weights.append(attn) # attn shape: [B, num_heads, seq_len, seq_len]

    # Extract CLS token feature for final normalization and classification
    cls_feat = self.norm(x[:, 0])
    out = self.head(cls_feat)

    return out, attn_weights