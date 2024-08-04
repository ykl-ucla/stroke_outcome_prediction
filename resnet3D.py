import torch
import torch.nn as nn
import torch.nn.functional as F

# Frozen BatchNorm Layer
class FrozenBN(nn.Module):
    def __init__(self, num_channels, momentum=0.1, eps=1e-5):
        super(FrozenBN, self).__init__()
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps
        self.params_set = False

    def set_params(self, scale, bias, running_mean, running_var):
        self.register_buffer('scale', scale)
        self.register_buffer('bias', bias)
        self.register_buffer('running_mean', running_mean)
        self.register_buffer('running_var', running_var)
        self.params_set = True

    def forward(self, x):
        assert self.params_set, 'model.set_params(...) must be called before the forward pass'
        return torch.batch_norm(x, self.scale, self.bias, self.running_mean, self.running_var, False, self.momentum, self.eps, torch.backends.cudnn.enabled)

    def __repr__(self):
        return f'FrozenBN({self.num_channels})'

def freeze_bn(m):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if isinstance(target_attr, nn.BatchNorm3d):
            frozen_bn = FrozenBN(target_attr.num_features, target_attr.momentum, target_attr.eps)
            frozen_bn.set_params(target_attr.weight.data, target_attr.bias.data, target_attr.running_mean, target_attr.running_var)
            setattr(m, attr_str, frozen_bn)
    for n, ch in m.named_children():
        freeze_bn(ch)

# Bottleneck Block for ResNet
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride, downsample, temp_conv, temp_stride):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1 + temp_conv * 2, 1, 1), 
                               stride=(temp_stride, 1, 1), padding=(temp_conv, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), 
                               stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# I3Res50forStrokeOutcome Model Definition
class I3Res50forStrokeOutcome(nn.Module):
    def __init__(self, input_cha=2, block=Bottleneck, layers=[3, 4, 6], num_classes=400):
        super(I3Res50forStrokeOutcome, self).__init__()
        self.inplanes = 64
        self.conv1_ = nn.Conv3d(input_cha, 64, kernel_size=(5, 7, 7), 
                                stride=(2, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, 
                                       temp_conv=[1, 1, 1], temp_stride=[1, 1, 1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, 
                                       temp_conv=[1, 0, 1, 0], temp_stride=[1, 1, 1, 1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, 
                                       temp_conv=[1, 0, 1, 0, 1, 0], temp_stride=[1, 1, 1, 1, 1, 1])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1_ = nn.Linear(512 * round(block.expansion / 2), 1)

        self.drop = nn.Dropout(0.80)
        self.drop3D = nn.Dropout3d(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, temp_conv, temp_stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or temp_stride[0] != 1:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1), 
                          stride=(temp_stride[0], stride, stride), padding=(0, 0, 0), bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, temp_conv[0], temp_stride[0]))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, temp_conv[i], temp_stride[i]))

        return nn.Sequential(*layers)

    def forward_single(self, x):
        x = self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.maxpool2(x)
        
        x = self.layer2(x)
        x = self.drop3D(x)
        
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.drop(x)
        
        x = x.view(x.shape[0], -1)
        x = self.fc1_(x)
        return x

    def forward(self, batch):
        return self.forward_single(batch)

# Function to initialize the model
def i3_res50forStrokeOutcome(input_cha, num_classes):
    net = I3Res50forStrokeOutcome(input_cha=input_cha, num_classes=num_classes)
    pretrained_dict = torch.load('pretrained/i3d_r50_kinetics.pth')
    model_dict = net.state_dict()
    overlap_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(overlap_dict)
    net.load_state_dict(model_dict)
    return net
