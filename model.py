"""
Modular U-Net Architecture for Retinal Vessel Segmentation.
Features batch normalization, robust skip connection tensor padding,
configurable channel depth, and optional bilinear/transposed conv upsampling.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    [ Conv2D -> BatchNorm -> ReLU ] x 2 with optional spatial dropout.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        dropout_rate: float = 0.0
    ):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))

        layers.extend([
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ])

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling stage: MaxPool2d (2x2) followed by DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling stage: Upsampling (Transposed Conv or Bilinear) followed by
    robust skip connection concatenation (with adaptive padding) and DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super(Up, self).__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Feature map from deeper layer to upsample
            x2: Skip-connection feature map from encoder
        """
        x1 = self.up(x1)

        # Robust tensor shape handling: adjust for rounding discrepancies on odd input dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_x != 0 or diff_y != 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
            )

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 Convolution mapping feature channels to class logits.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Production-grade U-Net for Retinal Vessel Segmentation.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        bilinear: bool = False,
        dropout_rate: float = 0.0,
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        c = base_channels
        factor = 2 if bilinear else 1

        # Encoder path
        self.inc = DoubleConv(in_channels, c, dropout_rate=dropout_rate)
        self.down1 = Down(c, c * 2, dropout_rate=dropout_rate)
        self.down2 = Down(c * 2, c * 4, dropout_rate=dropout_rate)
        self.down3 = Down(c * 4, c * 8, dropout_rate=dropout_rate)
        self.down4 = Down(c * 8, (c * 16) // factor, dropout_rate=dropout_rate)

        # Decoder path with skip connections
        self.up1 = Up(c * 16, (c * 8) // factor, bilinear)
        self.up2 = Up(c * 8, (c * 4) // factor, bilinear)
        self.up3 = Up(c * 4, (c * 2) // factor, bilinear)
        self.up4 = Up(c * 2, c, bilinear)

        # Final projection head
        self.outc = OutConv(c, out_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming (He) normal initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder forward pass with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing UNet architecture and forward pass...")
    model = UNet(in_channels=3, out_channels=1, base_channels=64)
    print(f"Total Trainable Parameters: {model.count_parameters():,}")

    # Standard input shape test
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    print(f"Standard Input [2, 3, 512, 512] -> Output Shape: {output.shape}")
    assert output.shape == (2, 1, 512, 512), "Output shape mismatch!"

    # Robust tensor shape handling test with odd dimensions
    odd_input = torch.randn(1, 3, 511, 511)
    odd_output = model(odd_input)
    print(f"Odd Dimensions [1, 3, 511, 511] -> Output Shape: {odd_output.shape}")
    assert odd_output.shape == (1, 1, 511, 511), "Odd shape mismatch!"

    print("UNet architecture tests PASSED successfully.")
