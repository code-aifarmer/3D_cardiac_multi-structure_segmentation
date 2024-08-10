import torch
from torch import nn
from torchsummary import summary
from graphviz import Digraph


class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Conv3d(channel, channel // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(channel // ratio, channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer(x)
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


# 创建网络实例
model = UNet(in_channels=1, out_channels=1)

# 将模型移动到GPU上
device = torch.device('cuda:0')
model.to(device)

# 打印网络结构信息
summary(model, (1, 64, 64, 64))  # 输入大小根据实际情况设定

# 绘制网络示意图
dot = Digraph(comment='UNet', format='png', graph_attr={'rankdir': 'LR'})

# 定义用于递归添加节点的函数
def add_nodes(dot, module, prefix):
    # 添加当前模块的节点
    dot.node(prefix, str(type(module).__name__), shape='box')

    # 递归添加子模块的节点
    for name, submodule in module.named_children():
        add_nodes(dot, submodule, f"{prefix}_{name}")

    # 添加当前模块与子模块的边
    for name, submodule in module.named_children():
        dot.edge(prefix, f"{prefix}_{name}")

# 添加网络结构节点
add_nodes(dot, model, "model")

# 调整节点和边的样式
dot.node_attr.update({'style': 'filled', 'fillcolor': 'lightblue'})
dot.edge_attr.update({'style': 'solid'})

# 保存示意图
dot.render('unet_structure', view=True)
