import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# LeNet-5 for FMNIST (1x28x28 -> 10 classes)
class LeNet(nn.Module):

    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Tiny CNN baseline for CIFAR/SVHN (3x32x32 -> 10 classes) 
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 32x16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 64x8x8
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

#WideResNet basic block
class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.equal_in_out = (in_planes == out_planes)
        self.conv_shortcut = (not self.equal_in_out) and nn.Conv2d(in_planes, out_planes, 1, stride, bias=False) or nn.Identity()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, padding=1, bias=False)

        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = x if self.equal_in_out else self.conv_shortcut(out)

        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)

        return out + shortcut

# Stacking N BasicBlocks
class NetworkBlock(nn.Module):
    def __init__(self, num_layers, in_planes, out_planes, stride, drop_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            s = stride if i == 0 else 1
            inp = in_planes if i == 0 else out_planes
            layers.append(BasicBlock(inp, out_planes, s, drop_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# WideResNet-D for CIFAR/SVHN/CIFAR-100
# Depth = 6n+4, Widen factor k, dropout rate, and num_classes
class WideResNet(nn.Module):

    def __init__(self, depth=28, widen_factor=2, drop_rate=0.0, num_classes=10, in_channels=3):
        super().__init__()
        assert (depth - 4) % 6 == 0, "Depth must be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor

        n_channels = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = nn.Conv2d(in_channels, n_channels[0], 3, 1, padding=1, bias=False)

        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], stride=1, drop_rate=drop_rate)
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], stride=2, drop_rate=drop_rate)
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], stride=2, drop_rate=drop_rate)

        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.fc(out)

# name: lenet, simplecnn, wrn-28-2, wrn-28-8
def get_model(name: str, num_classes: int, **kwargs) -> nn.Module:

    if name.lower() == 'lenet':
        return LeNet(num_classes=num_classes, **kwargs)
    if name.lower() == 'simplecnn':
        return SimpleCNN(num_classes=num_classes, **kwargs)
    if name.lower().startswith('wrn'):
        #wrn-28-2 or wrn-28-8
        parts = name.split('-')
        depth = int(parts[1])
        wf = int(parts[2])
        return WideResNet(depth=depth, widen_factor=wf, num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model name: {name}")
