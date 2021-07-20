import torch
from torch import nn

from torch.nn import GRUCell

class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [16, 32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.spatial2_0 = VGGSpatialBlock(nb_filter[1], nb_filter[2], nb_filter[2], False)
        
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.spatial3_0 = VGGSpatialBlock(nb_filter[2], nb_filter[3], nb_filter[3], False)

        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.spatial4_0 = VGGSpatialBlock(nb_filter[3], nb_filter[4], nb_filter[4], False)

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.spatial2_1 = VGGSpatialBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2], True)

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.spatial3_1 = VGGSpatialBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3], True)

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        self.spatial2_2 = VGGSpatialBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2], True)

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        #x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0 = self.spatial2_0(self.pool(x1_0))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        #x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0 = self.spatial3_0(self.pool(x2_0))

        #x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x2_1 = self.spatial2_1(torch.cat([x2_0, self.up(x3_0)], 1))

        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        #x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0 = self.spatial4_0(self.pool(x3_0))
        
        #x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x3_1 = self.spatial3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        #x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x2_2 = self.spatial2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [16, 32, 64, 128, 256, 512]
        
        down_upsample = False
        up_upsample = True

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.spatial0_0 = VGGSpatialBlock(input_channels, nb_filter[0], nb_filter[0], down_upsample)
        
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.spatial1_0 = VGGSpatialBlock(nb_filter[0], nb_filter[1], nb_filter[1], down_upsample)
        
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.spatial2_0 = VGGSpatialBlock(nb_filter[1], nb_filter[2], nb_filter[2], down_upsample)

        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.spatial3_0 = VGGSpatialBlock(nb_filter[2], nb_filter[3], nb_filter[3], down_upsample)
        
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.spatial4_0 = VGGSpatialBlock(nb_filter[3], nb_filter[4], nb_filter[4], down_upsample)
        
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.spatial3_1 = VGGSpatialBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3], up_upsample)
        
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.spatial2_2 = VGGSpatialBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2], up_upsample)
        
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.spatial1_3 = VGGSpatialBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1], up_upsample)

        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.spatial0_4 = VGGSpatialBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0], up_upsample)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        #x0_0 = self.conv0_0(input)
        x0_0 = self.spatial0_0(input)
        
        x1_0 = self.conv1_0(self.pool(x0_0))
        #x1_0 = self.spatial1_0(self.pool(x0_0))

        #x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0 = self.spatial2_0(self.pool(x1_0))
        
        #x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0 = self.spatial3_0(self.pool(x2_0))
        
        #x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0 = self.spatial4_0(self.pool(x3_0))
        
        #x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x3_1 = self.spatial3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        
        #x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x2_2 = self.spatial2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1)) 
        #x1_3 = self.spatial1_3(torch.cat([x1_0, self.up(x2_2)], 1))  

        #x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        x0_4 = self.spatial0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        
        output = self.final(x0_4)
        return output

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
    
class VGGSpatialBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, upsample=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        
        self.spatial = ResSDNLayer(middle_channels, out_channels, 16, range(4), 3, 1, 1, upsample)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #out = self.conv2(out)
        out = self.spatial(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class SDNCell(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.gru = GRUCell(input_size=3*num_features, hidden_size=num_features)

    def forward(self, neighboring_features, features):
        """ Update current features based on neighboring features """
        return self.gru(torch.cat(neighboring_features, dim=1), features)


class _CorrectionLayer(nn.Module):

    def __init__(self, num_features, dir=0):
        super().__init__()
        self.num_features = num_features
        self.cell = SDNCell(num_features)
        if dir == 0:
            self.forward = self.forward0
        elif dir == 1:
            self.forward = self.forward1
        elif dir == 2:
            self.forward = self.forward2
        else:
            self.forward = self.forward3
        self.zero_pad = None

    def _get_zero_pad(self, batch, device):
        if self.zero_pad is None or self.zero_pad.shape[0] != batch:
            self.zero_pad = torch.zeros((batch, self.num_features, 1), device=device)  # no grad ??
        return self.zero_pad

    def forward0(self, features):
        batch = features.shape[0]
        dim = features.shape[2]
        zero_pad = self._get_zero_pad(batch, features.device)
        # make a loop
        for d in range(1, dim):
            neighboring_features = torch.cat([zero_pad, features[:, :, :, d - 1], zero_pad], dim=2).transpose(1, 2)
            # compute features
            features[:, :, :, d] = self.cell(
                neighboring_features=[neighboring_features[:, :-2, :].reshape(-1, self.num_features),
                                      neighboring_features[:, 1:-1, :].reshape(-1, self.num_features),
                                      neighboring_features[:, 2:, :].reshape(-1, self.num_features)],
                features=features[:, :, :, d].transpose(1, 2).reshape(-1, self.num_features).clone()
            ).reshape(batch, -1, self.num_features).transpose(1, 2)
        # return new features
        return features

    def forward1(self, features):
        batch = features.shape[0]
        dim = features.shape[2]
        zero_pad = self._get_zero_pad(batch, features.device)
        # make a loop
        for d in range(dim - 2, -1, -1):
            neighboring_features = torch.cat([zero_pad, features[:, :, :, d + 1], zero_pad], dim=2).transpose(1, 2)
            # compute features
            features[:, :, :, d] = self.cell(
                neighboring_features=[neighboring_features[:, :-2, :].reshape(-1, self.num_features),
                                      neighboring_features[:, 1:-1, :].reshape(-1, self.num_features),
                                      neighboring_features[:, 2:, :].reshape(-1, self.num_features)],
                features=features[:, :, :, d].transpose(1, 2).reshape(-1, self.num_features).clone()
            ).reshape(batch, -1, self.num_features).transpose(1, 2)
        # return new features
        return features

    def forward2(self, features):
        batch = features.shape[0]
        dim = features.shape[2]
        zero_pad = self._get_zero_pad(batch, features.device)
        # make a loop
        for d in range(1, dim):
            neighboring_features = torch.cat([zero_pad, features[:, :, d - 1, :], zero_pad], dim=2).transpose(1, 2)
            # compute features
            features[:, :, d, :] = self.cell(
                neighboring_features=[neighboring_features[:, :-2, :].reshape(-1, self.num_features),
                                      neighboring_features[:, 1:-1, :].reshape(-1, self.num_features),
                                      neighboring_features[:, 2:, :].reshape(-1, self.num_features)],
                features=features[:, :, d, :].transpose(1, 2).reshape(-1, self.num_features).clone()
            ).reshape(batch, -1, self.num_features).transpose(1, 2)
        # return new features
        return features

    def forward3(self, features):
        batch = features.shape[0]
        dim = features.shape[2]
        zero_pad = self._get_zero_pad(batch, features.device)
        # make a loop
        for d in range(dim - 2, -1, -1):
            neighboring_features = torch.cat([zero_pad, features[:, :, d + 1, :], zero_pad], dim=2).transpose(1, 2)
            # compute features
            features[:, :, d, :] = self.cell(
                neighboring_features=[neighboring_features[:, :-2, :].reshape(-1, self.num_features),
                                      neighboring_features[:, 1:-1, :].reshape(-1, self.num_features),
                                      neighboring_features[:, 2:, :].reshape(-1, self.num_features)],
                features=features[:, :, d, :].transpose(1, 2).reshape(-1, self.num_features).clone()
            ).reshape(batch, -1, self.num_features).transpose(1, 2)
        # return new features
        return features


class SDNLayer(nn.Module):
    def __init__(self, in_ch, out_ch, num_features, dirs, kernel_size, stride, padding, upsample):
        super().__init__()
        # project-in network
        cnn_module = nn.ConvTranspose2d if 2 else nn.Conv2d
        self.project_in_stage = cnn_module(in_ch, num_features, kernel_size, stride, padding)
        # correction network
        sdn_correction_layers = []
        for dir in dirs:
            sdn_correction_layers.append(_CorrectionLayer(num_features, dir=dir))
        self.sdn_correction_stage = nn.Sequential(*sdn_correction_layers)
        # project-out network
        self.project_out_stage = nn.Conv2d(num_features, out_ch, 1)

    def forward(self, x):
        # (I) project-in stage
        x = self.project_in_stage(x)
        x = torch.tanh(x)
        # (II) correction stage
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.sdn_correction_stage(x)
        x = x.contiguous(memory_format=torch.contiguous_format)
        # (III) project-out stage
        x = self.project_out_stage(x)
        return x


class ResSDNLayer(nn.Module):

    def __init__(self, in_ch, out_ch, num_features, dirs, kernel_size, stride, padding, upsample):
        super().__init__()
        self.sdn = SDNLayer(in_ch, 2 * out_ch, num_features, dirs, kernel_size, stride, padding, upsample)
        cnn_module = nn.ConvTranspose2d if upsample else nn.Conv2d
        self.cnn = cnn_module(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, input):
        cnn_out = self.cnn(input)
        sdn_out, gate = self.sdn(input).chunk(2, 1)
        gate = torch.sigmoid(gate)
        return gate * cnn_out + (1-gate) * sdn_out