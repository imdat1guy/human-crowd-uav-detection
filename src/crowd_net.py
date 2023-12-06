import torch
import torch.nn as nn
import numpy as np
import math

def create_conv2d_block(in_channels, kernel_size, n_filter, dilated_rate):
    """
    o = output
    p = padding
    k = kernel_size
    s = stride
    d = dilation
    """
    k = kernel_size
    d = dilated_rate
    padding_rate = int((k + (k-1)*(d-1))/2)
    conv2d =  nn.Conv2d(in_channels, n_filter, kernel_size, padding=padding_rate, dilation = dilated_rate)
    bn = nn.BatchNorm2d(n_filter)
    relu = nn.ReLU(inplace=True)
    return [conv2d, bn, relu]

def make_layers_by_cfg(cfg, in_channels = 3,batch_norm=True, dilation = True):
    """
    cfg: list of tuple (number of layer, kernel, n_filter, dilated) or 'M'
    """
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # number of layer, kernel, n_filter, dilated
            for t in range(v[0]):
              layers += create_conv2d_block(in_channels, v[1], v[2], v[3])
              in_channels = v[2]
    return nn.Sequential(*layers)



class CustomVGG16(nn.Module):
    def __init__(self, pretrain=None, logger=None):
        super(CustomVGG16, self).__init__()

        self.conv1_1 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(64, 128, (3, 3), stride=1, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, (3, 3), stride=1, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3_1 = nn.Conv2d(128, 256, (3, 3), stride=1, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv4_1 = nn.Conv2d(256, 512, (3, 3), stride=1, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)
        self.conv5_1 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_3 = nn.ReLU(inplace=True)
        if pretrain:
            if '.npy' in pretrain:
                state_dict = np.load(pretrain).item()
                for k in state_dict:
                    state_dict[k] = torch.from_numpy(state_dict[k])
            else:
                state_dict = torch.load(pretrain)
            own_state_dict = self.state_dict()
            for name, param in own_state_dict.items():
                if name in state_dict:
                    if logger:
                        logger.info('copy the weights of %s from pretrained model' % name)
                    param.copy_(state_dict[name])
                else:
                    if logger:
                        logger.info('init the weights of %s from mean 0, std 0.01 gaussian distribution'\
                         % name)
                    if 'bias' in name:
                        param.zero_()
                    else:
                        param.normal_(0, 0.01)
        else:
            self._initialize_weights(logger)

    def forward(self, x):

        conv1_1 = self.relu1_1(self.conv1_1(x))
        conv1_2 = self.relu1_2(self.conv1_2(conv1_1))

        conv2_1 = self.relu2_1(self.conv2_1(conv1_2))
        conv2_2 = self.relu2_2(self.conv2_2(conv2_1))

        conv3_1 = self.relu3_1(self.conv3_1(conv2_2))
        conv3_2 = self.relu3_2(self.conv3_2(conv3_1))
        conv3_3 = self.relu3_3(self.conv3_3(conv3_2))

        conv4_1 = self.relu4_1(self.conv4_1(conv3_3))
        conv4_2 = self.relu4_2(self.conv4_2(conv4_1))
        conv4_3 = self.relu4_3(self.conv4_3(conv4_2))
        pool4 = self.pool4(conv4_3)

        conv5_1 = self.relu5_1(self.conv5_1(conv4_3))
        conv5_2 = self.relu5_2(self.conv5_2(conv5_1))
        conv5_3 = self.relu5_3(self.conv5_3(conv5_2))

        side = [conv1_1, conv1_2, conv2_1, conv2_2,
                conv3_1, conv3_2, conv3_3, conv4_1,
                conv4_2, conv4_3, conv5_1, conv5_2, conv5_3]

        return side

    def _initialize_weights(self, logger=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if logger:
                        logger.info('init the weights of %s from mean 0, std 0.01 gaussian distribution'\
                         % m)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Scale(nn.Module):
    def __init__(self, c_in, rate):
        super(Scale, self).__init__()
        self.a = nn.Sequential(*create_conv2d_block(c_in, 3, c_in, rate*1))
        self.b = nn.Sequential(*create_conv2d_block(c_in, 3, c_in, rate*2))
        self.c = nn.Sequential(*create_conv2d_block(c_in, 3, c_in, rate*4))

    def forward(self,x):
        xa = self.a(x)
        xb = self.b(x)
        xc = self.c(x)

        out = xa + xb + xc
        return out

class BiDirectionalCascadeNet(nn.Module):
    def __init__(self, pretrain=None, logger=None, rate=5):
        super(BiDirectionalCascadeNet, self).__init__()
        self.pretrain = pretrain
        t = 1
        self.features = CustomVGG16(pretrain, logger)
        self.msblock1_1 = Scale(64,1)
        self.msblock1_2 = Scale(64,1)
        self.conv1_1_down = nn.Conv2d(32*2, 21, (3, 3), stride=1,padding=1)
        self.conv1_2_down = nn.Conv2d(32*2, 21, (3, 3), stride=1,padding=1)
        self.score_dsn1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn1_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock2_1 = Scale(128,2)
        self.msblock2_2 = Scale(128,2)
        self.conv2_1_down = nn.Conv2d(32*4,21, (3, 3), stride=1,padding=1)
        self.conv2_2_down = nn.Conv2d(32*4,21, (3, 3), stride=1,padding=1)
        self.score_dsn2 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn2_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock3_1 = nn.Sequential(Scale(256,4))
        self.msblock3_2 = nn.Sequential(Scale(256,4))
        self.msblock3_3 = nn.Sequential(Scale(256,4))
        self.conv3_1_down = nn.Conv2d(32*8, 21, (3, 3), stride=1,padding=1)
        self.conv3_2_down = nn.Conv2d(32*8, 21, (3, 3), stride=1,padding=1)
        self.conv3_3_down = nn.Conv2d(32*8, 21, (3, 3), stride=1,padding=1)
        self.score_dsn3 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn3_1 = nn.Conv2d(21, 1, (1, 1), stride=1)

        self.fuse = nn.Conv2d(6, 1, 1, stride=1)

    def forward(self, x):
        features = self.features(x)


        sum1 = self.conv1_1_down(self.msblock1_1(features[0])) + \
                self.conv1_2_down(self.msblock1_2(features[1]))
        s1 = self.score_dsn1(sum1)
        s11 = self.score_dsn1_1(sum1)

        sum2 = self.conv2_1_down(self.msblock2_1(features[2])) + \
            self.conv2_2_down(self.msblock2_2(features[3]))
        s2 = self.score_dsn2(sum2)
        s21 = self.score_dsn2_1(sum2)


        sum3 = self.conv3_1_down(self.msblock3_1(features[4])) + \
            self.conv3_2_down(self.msblock3_2(features[5])) + \
           self.conv3_3_down(self.msblock3_3(features[6]))
        s3 = self.score_dsn3(sum3)

        s31 = self.score_dsn3_1(sum3)

        fuse = self.fuse(torch.cat([s1,s11, s2,s21,s3,s31], 1))
        return fuse


class CrowdNet(nn.Module):
    def __init__(self):
        super(CrowdNet, self).__init__()

        self.frontend_config = [(2,3,64,1),(2,3,64,1),'M', (2,3,128,1),(2,3,128,1),'M', (2,3,256,1), (2,3,256,1), 'M',(2,3,512,1), (2,3,512,1),(2,3,64,1)]
        self.frontend = make_layers_by_cfg(self.frontend_config)
        self.vgg= BiDirectionalCascadeNet()
        self.seen = 0

    def forward(self,x):
        x = self.frontend(x)
        x = self.vgg(x)
        return x