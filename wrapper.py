import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any, Optional
import math


class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM) from CBAM"""

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM) from CBAM"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class MultiHeadSpatialAttention(nn.Module):
    """Multi-Head Spatial Attention for 2D feature maps"""

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadSpatialAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # Generate Q, K, V
        q = self.q_proj(x)  # [B, C, H, W]
        k = self.k_proj(x)  # [B, C, H, W]
        v = self.v_proj(x)  # [B, C, H, W]

        # Reshape for multi-head attention
        # [B, num_heads, head_dim, H*W]
        q = q.view(B, self.num_heads, self.head_dim, H * W)
        k = k.view(B, self.num_heads, self.head_dim, H * W)
        v = v.view(B, self.num_heads, self.head_dim, H * W)

        # Attention computation
        # [B, num_heads, H*W, H*W]
        attn = torch.matmul(q.transpose(-2, -1), k) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        # [B, num_heads, head_dim, H*W]
        out = torch.matmul(v, attn.transpose(-2, -1))

        # Reshape back to spatial dimensions
        out = out.view(B, C, H, W)
        out = self.out_proj(out)

        return out


class EfficientNetMultiHeadAttention(nn.Module):
    """EfficientNet-B3 with Multi-Head Attention for Fighting Game Feature Extraction"""

    def __init__(
        self,
        num_input_channels=27,  # 9 frames × 3 RGB channels
        features_dim=512,
        num_attention_heads=8,
        use_cbam=True,
        use_spatial_attention=True,
        dropout=0.1,
    ):
        super(EfficientNetMultiHeadAttention, self).__init__()

        self.num_input_channels = num_input_channels
        self.features_dim = features_dim
        self.use_cbam = use_cbam
        self.use_spatial_attention = use_spatial_attention

        # Load pre-trained EfficientNet-B3
        self.efficientnet = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )

        # Modify first conv layer to accept our input channels
        original_conv = self.efficientnet.features[0][0]
        self.efficientnet.features[0][0] = nn.Conv2d(
            num_input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # Remove the final classification layer
        self.efficientnet.classifier = nn.Identity()

        # Get the number of features from EfficientNet-B3 (1536)
        efficientnet_features = 1536

        # Add CBAM attention to intermediate features if requested
        if self.use_cbam:
            # Add CBAM at multiple stages for better feature refinement
            self.cbam_blocks = nn.ModuleList(
                [
                    CBAM(48),  # After block 2
                    CBAM(80),  # After block 3
                    CBAM(160),  # After block 5
                    CBAM(224),  # After block 6
                ]
            )

        # Multi-Head Spatial Attention
        if self.use_spatial_attention:
            self.spatial_attention = MultiHeadSpatialAttention(
                embed_dim=efficientnet_features,
                num_heads=num_attention_heads,
                dropout=dropout,
            )

        # Feature fusion and projection
        self.feature_projection = nn.Sequential(
            nn.Conv2d(efficientnet_features, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, features_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        print(f"🧠 EFFICIENTNET-B3 + MULTI-HEAD ATTENTION:")
        print(f"   📊 Input channels: {num_input_channels}")
        print(f"   🎯 Output features: {features_dim}")
        print(f"   👁️ Attention heads: {num_attention_heads}")
        print(f"   🔍 CBAM enabled: {use_cbam}")
        print(f"   🎨 Spatial attention: {use_spatial_attention}")
        print(f"   💫 Optimized for fighting game pattern recognition")

    def forward(self, x):
        # Input normalization (EfficientNet expects [0,1] range)
        x = x.float() / 255.0

        features = []

        # Pass through EfficientNet backbone with intermediate feature extraction
        for i, layer in enumerate(self.efficientnet.features):
            x = layer(x)

            # Collect features for CBAM attention at specific layers
            if self.use_cbam:
                if i == 2:  # After block 2 (48 channels)
                    x = self.cbam_blocks[0](x)
                elif i == 3:  # After block 3 (80 channels)
                    x = self.cbam_blocks[1](x)
                elif i == 5:  # After block 5 (160 channels)
                    x = self.cbam_blocks[2](x)
                elif i == 6:  # After block 6 (224 channels)
                    x = self.cbam_blocks[3](x)

        # Average pooling as in original EfficientNet
        x = self.efficientnet.avgpool(x)

        # Apply Multi-Head Spatial Attention if enabled
        if self.use_spatial_attention and x.size(2) > 1 and x.size(3) > 1:
            # Only apply if we have spatial dimensions > 1x1
            x = self.spatial_attention(x)

        # Final feature projection
        features = self.feature_projection(x)

        return features

    def get_attention_maps(self, x):
        """Extract attention maps for visualization"""
        x = x.float() / 255.0
        attention_maps = {}

        # CBAM attention maps
        if self.use_cbam:
            for i, layer in enumerate(self.efficientnet.features):
                x = layer(x)

                if i == 2:
                    ca_map = self.cbam_blocks[0].channel_attention(x)
                    sa_map = self.cbam_blocks[0].spatial_attention(x)
                    attention_maps["cbam_block2"] = {
                        "channel": ca_map,
                        "spatial": sa_map,
                    }
                    x = self.cbam_blocks[0](x)
                elif i == 3:
                    ca_map = self.cbam_blocks[1].channel_attention(x)
                    sa_map = self.cbam_blocks[1].spatial_attention(x)
                    attention_maps["cbam_block3"] = {
                        "channel": ca_map,
                        "spatial": sa_map,
                    }
                    x = self.cbam_blocks[1](x)
                elif i == 5:
                    ca_map = self.cbam_blocks[2].channel_attention(x)
                    sa_map = self.cbam_blocks[2].spatial_attention(x)
                    attention_maps["cbam_block5"] = {
                        "channel": ca_map,
                        "spatial": sa_map,
                    }
                    x = self.cbam_blocks[2](x)
                elif i == 6:
                    ca_map = self.cbam_blocks[3].channel_attention(x)
                    sa_map = self.cbam_blocks[3].spatial_attention(x)
                    attention_maps["cbam_block6"] = {
                        "channel": ca_map,
                        "spatial": sa_map,
                    }
                    x = self.cbam_blocks[3](x)

        x = self.efficientnet.avgpool(x)

        # Multi-head spatial attention maps would require modification to the attention module
        # to return attention weights - omitted for brevity

        return attention_maps


class EfficientNetB3FeatureExtractor(EfficientNetMultiHeadAttention):
    """Simplified interface matching the original DeepCNNFeatureExtractor"""

    def __init__(self, observation_space, features_dim: int = 512):
        # Extract input channels from observation space
        num_input_channels = observation_space.shape[0]

        super().__init__(
            num_input_channels=num_input_channels,
            features_dim=features_dim,
            num_attention_heads=8,
            use_cbam=True,
            use_spatial_attention=True,
            dropout=0.1,
        )

        print(f"🚀 FIGHTING GAME FEATURE EXTRACTOR:")
        print(f"   📊 Observation space: {observation_space.shape}")
        print(f"   🎯 Features output: {features_dim}")
        print(f"   🧠 Architecture: EfficientNet-B3 + Multi-Head Attention + CBAM")
        print(f"   ⚔️ Optimized for fighting game temporal patterns")


# Example usage and testing
if __name__ == "__main__":
    # Test the feature extractor
    import gymnasium as gym

    # Simulate fighting game observation space (27 channels, 126x180)
    observation_space = gym.spaces.Box(
        low=0, high=255, shape=(27, 126, 180), dtype="uint8"
    )

    # Create the feature extractor
    feature_extractor = EfficientNetB3FeatureExtractor(
        observation_space=observation_space, features_dim=512
    )

    # Test with dummy input
    batch_size = 4
    dummy_input = torch.randint(0, 256, (batch_size, 27, 126, 180), dtype=torch.float32)

    print(f"\n🔬 TESTING:")
    print(f"   Input shape: {dummy_input.shape}")

    with torch.no_grad():
        features = feature_extractor(dummy_input)
        print(f"   Output shape: {features.shape}")
        print(f"   ✅ Feature extraction successful!")

        # Test attention map extraction
        attention_maps = feature_extractor.get_attention_maps(dummy_input[:1])
        print(f"   📊 Attention maps extracted: {len(attention_maps)} layers")

    # Memory usage estimation
    total_params = sum(p.numel() for p in feature_extractor.parameters())
    trainable_params = sum(
        p.numel() for p in feature_extractor.parameters() if p.requires_grad
    )

    print(f"\n📊 MODEL STATISTICS:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / (1024**2):.1f} MB (FP32)")
    print(f"   🎯 Ready for fighting game RL training!")
