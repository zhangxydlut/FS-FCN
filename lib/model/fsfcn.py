import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision


affine_par = True


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
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


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts, channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts, channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts, 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts, 1)

    return weight_splits, bias_splits, num_insts


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:#注意，resnet中每个大block内的第一个bottleneck之后又个downsample
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x2 = x
        x = self.layer3(x)
        # x = self.layer4(x)
        x = torch.cat([x2, x], dim=1)

        x = self.layer5(x)
        return x


class FSFCN(nn.Module):
    def __init__(self, block, layers):
        super(FSFCN, self).__init__()

        self.backbone = ResNet(block, layers)
        self.load_resnet50_param(self.backbone, stop_layer='layer4', )
        self.build_dynamid_mask_head()

        self._init_layers()

    def load_resnet50_param(self, model, stop_layer='layer4'):
        resnet50 = torchvision.models.resnet50(pretrained=True)
        saved_state_dict = resnet50.state_dict()
        new_params = model.state_dict().copy()

        for i in saved_state_dict:  # copy params from resnet50,except layers after stop_layer
            i_parts = i.split('.')
            if not i_parts[0] == stop_layer:
                new_params['.'.join(i_parts)] = saved_state_dict[i]
            else:
                break
        model.load_state_dict(new_params)
        return model

    def build_dynamid_mask_head(self):
        # CondInst style dynamic conv FCN
        num_conv = 4
        self.dim_cond = 153
        self.dim_dynamic_conv = 8
        self.weight_nums = [64, 64, 8]
        self.bais_nums = [8, 8, 1]
        towers = [
            nn.Sequential(
                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()) for _ in range(num_conv)]

        self.controller_generator = nn.Sequential(
            *towers,
            nn.Conv2d(256, self.dim_cond, kernel_size=3, stride=1, padding=1))


        query_towers = [
            nn.Sequential(
                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()) for _ in range(num_conv)]

        self.query_tower_generator = nn.Sequential(
            *query_towers,
            nn.Conv2d(256, self.dim_dynamic_conv, kernel_size=3, stride=1, padding=1))

    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def dynamic_conv(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        feat = []
        for i in range(num_insts):
            n_layers = len(weights)
            x = features[i][None]
            for j, (w, b) in enumerate(zip(weights, biases)):
                x = F.conv2d(x, w[i], bias=b[i], stride=1, padding=0)
                if j < n_layers - 1:
                    x = F.relu(x)
            feat.append(x)
        return torch.cat(feat, dim=0)

    def forward(self, query_rgb, support_rgb, support_mask):
        # important: do not optimize the RESNET backbone
        query_feat = self.backbone(query_rgb).detach()  # freeze backbone
        support_feat = self.backbone(support_rgb).detach()

        # prepare dynamic kernel
        support_mask = F.interpolate(support_mask, support_feat.shape[-2:], mode='bilinear',align_corners=True)
        z = support_mask * support_feat
        z = self.controller_generator(z)
        z_reduced = F.avg_pool2d(z, z.shape[-2:]).reshape(z.shape[0], -1)
        weight_splits, bias_splits, num_inst = parse_dynamic_params(
            z_reduced, 8, self.weight_nums, self.bais_nums)

        # prepare query feat
        query_tower = self.query_tower_generator(query_feat)

        # get segmentation
        out = self.dynamic_conv(query_tower, weight_splits, bias_splits, num_inst)
        return out


def build_model():
    model = FSFCN(Bottleneck, [3, 4, 6, 3])
    return model
