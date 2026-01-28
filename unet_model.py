# coding:utf-8
from torch import nn
import torch
from iresnet import iresnet50

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        #
        # return x

    def forward(self, x):
        return self._forward_impl(x)

class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        self.resnet50 = iresnet50()
        # self.resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
        # self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        # self.up_conv1 = self._make_up_conv_layer(2048, 1024)
        # self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # self.up_conv2 = self._make_up_conv_layer(1024, 512)
        # self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # self.up_conv3 = self._make_up_conv_layer(512, 256)
        # self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.up_conv4 = self._make_up_conv_layer(128, 128)
        # self.up5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.up_conv5 = self._make_up_conv_layer(64, 32)
        # self.conv = nn.Conv2d(32, 3, kernel_size=1, bias=False)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = self._make_up_conv_layer(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = self._make_up_conv_layer(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = self._make_up_conv_layer(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up_conv4 = self._make_up_conv_layer(64, 32)
        # self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        # self.up_conv5 = self._make_up_conv_layer(64, 32)
        self.conv = nn.Conv2d(32, 3, kernel_size=1, bias=False)

    def _make_up_conv_layer(self, in_channels, out_channels):
        mid_channels = in_channels // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.resnet50(x)
        y3 = self.up_conv1(torch.cat([self.up1(x4), x3], dim=1))
        y2 = self.up_conv2(torch.cat([self.up2(y3), x2], dim=1))
        y1 = self.up_conv3(torch.cat([self.up3(y2), x1], dim=1))
        y = self.up_conv4(self.up4(y1))
        # y = self.up_conv5(self.up5(y))
        y = self.conv(y)
        return y


class Discriminator512(nn.Module):
    def __init__(self, input_channels=3, ndf=64):
        super(Discriminator512, self).__init__()
        
        # 序列1: 512x512 -> 256x256
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 序列2: 256x256 -> 128x128
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 序列3: 128x128 -> 64x64
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 序列4: 64x64 -> 32x32
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 序列5: 32x32 -> 16x16
        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf*8, ndf*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 序列6: 16x16 -> 8x8
        self.conv6 = nn.Sequential(
            nn.Conv2d(ndf*16, ndf*32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 序列7: 8x8 -> 4x4
        self.conv7 = nn.Sequential(
            nn.Conv2d(ndf*32, ndf*64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 输出层: 4x4 -> 1x1
        self.output = nn.Sequential(
            nn.Conv2d(ndf*64, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        # 前向传播
        x = self.conv1(x)  # 256x256
        x = self.conv2(x)  # 128x128
        x = self.conv3(x)  # 64x64
        x = self.conv4(x)  # 32x32
        x = self.conv5(x)  # 16x16
        x = self.conv6(x)  # 8x8
        x = self.conv7(x)  # 4x4
        x = self.output(x) # 1x1
        
        return x.view(-1)  # 展平输出


# model = UNet()
# x = torch.randn(4, 3, 512, 512)
# y = model(x)
# print(y.shape)