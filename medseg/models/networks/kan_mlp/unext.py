"""UNeXt - lightweight UNet with tokenized MLP bottleneck."""
# Source: https://github.com/jeya-maria-jose/UNeXt-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class TokenizedMLP(nn.Module):
    """Tokenized MLP block from UNeXt: shift MLP for spatial mixing."""
    def __init__(self, dim, shift_size=5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.dwconv = nn.Conv1d(dim * 2, dim * 2, shift_size, padding=shift_size // 2, groups=dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # B, N, C
        x_res = x
        x = self.norm(x)
        x = self.fc1(x)
        x = x.permute(0, 2, 1)  # B, 2C, N
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # B, N, 2C
        x = self.act(x)
        x = self.fc2(x)
        x = x + x_res
        return x.permute(0, 2, 1).reshape(B, C, H, W)


class UNeXt(nn.Module):
    """UNeXt: lightweight UNet with shifted MLP bottleneck.

    Architecture: Conv encoder -> Tokenized MLP bottleneck -> Conv decoder
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 base_ch=32, num_mlp_blocks=3, deep_supervision=False, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        # Encoder (lightweight convolutions)
        self.enc1 = nn.Sequential(ConvBlock(in_channels, chs[0]), ConvBlock(chs[0], chs[0]))
        self.enc2 = nn.Sequential(ConvBlock(chs[0], chs[1]), ConvBlock(chs[1], chs[1]))
        self.enc3 = nn.Sequential(ConvBlock(chs[1], chs[2]), ConvBlock(chs[2], chs[2]))
        self.pool = nn.MaxPool2d(2)

        # Tokenized MLP bottleneck
        self.bottleneck_proj = ConvBlock(chs[2], chs[3])
        self.mlp_blocks = nn.ModuleList([TokenizedMLP(chs[3]) for _ in range(num_mlp_blocks)])
        self.bottleneck_pool = nn.MaxPool2d(2)

        # Decoder
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = nn.Sequential(ConvBlock(chs[2] * 2, chs[2]), ConvBlock(chs[2], chs[2]))
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = nn.Sequential(ConvBlock(chs[1] * 2, chs[1]), ConvBlock(chs[1], chs[1]))
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = nn.Sequential(ConvBlock(chs[0] * 2, chs[0]), ConvBlock(chs[0], chs[0]))

        self.head = nn.Conv2d(chs[0], num_classes, 1)

        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(chs[2], num_classes, 1),
                nn.Conv2d(chs[1], num_classes, 1),
            ])

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck with tokenized MLP
        b = self.bottleneck_proj(self.bottleneck_pool(e3))
        for mlp in self.mlp_blocks:
            b = mlp(b)

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.head(d1)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        if self.training and self.deep_supervision:
            input_size = out.shape[2:]
            aux = []
            for feat, head in zip([d3, d2], self.ds_heads):
                a = head(feat)
                if a.shape[2:] != input_size:
                    a = F.interpolate(a, size=input_size, mode='bilinear',
                                      align_corners=False)
                aux.append(a)
            return [out] + aux
        return out
