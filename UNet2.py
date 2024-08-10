import torch
from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(num_features=in_channels // 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels=in_channels // 4, out_channels=in_channels // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(num_features=in_channels // 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels=in_channels // 4, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.layer(x)
        return x * attn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention(out_channels)
        self.dropout = nn.Dropout3d(p=0.3)

    def forward(self, x):
        x = self.layer(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.dropout(x)
        return x

class DecBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecBlock, self).__init__()
        self.down = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, y):
        x = self.down(x)
        x = torch.cat([y, x], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, root_feat_maps=16, pool_size=2):
        super(UNet, self).__init__()
        self.first = ConvBlock(in_channels, root_feat_maps)
        self.down1 = nn.Sequential(
            nn.MaxPool3d(pool_size),
            ConvBlock(root_feat_maps, root_feat_maps * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool3d(pool_size),
            ConvBlock(root_feat_maps * 2, root_feat_maps * 4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool3d(pool_size),
            ConvBlock(root_feat_maps * 4, root_feat_maps * 8)
        )
        self.up1 = DecBlock(root_feat_maps * 8, root_feat_maps * 4)
        self.up2 = DecBlock(root_feat_maps * 4, root_feat_maps * 2)
        self.up3 = DecBlock(root_feat_maps * 2, root_feat_maps)
        self.final = nn.Conv3d(root_feat_maps, out_channels, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out1 = self.first(x)
        out2 = self.down1(out1)
        out3 = self.down2(out2)
        out4 = self.down3(out3)
        out = self.up1(out4, out3)
        out = self.up2(out, out2)
        out = self.up3(out, out1)
        out = self.final(out)
        return out
