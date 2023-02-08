import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt


class simpleNet_patch(nn.Module):
    def __init__(self, Y=True):
        super(simpleNet_patch, self).__init__()
        d = 1
        if Y == False:
            d = 3
        self.input = nn.Conv2d(in_channels=d, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=128, out_channels=d, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.cousin_input = nn.Conv2d(in_channels=d, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.cousin_conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.cousin_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.cousin_conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.cousin_conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.cousin_conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.cousin_conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x, y):
        residual = x
        inputs = self.input(self.relu(x))
        out = inputs
        out = self.conv1(self.relu(out))
        out = self.conv2(self.relu(out))
        out = self.conv3(self.relu(out))
        out = self.conv4(self.relu(out))
        out = self.conv5(self.relu(out))
        out = self.conv6(self.relu(out))

        cousin_input = self.cousin_input(self.relu(y))
        cousin_res = cousin_input
        cousin_out = cousin_input
        cousin_out = self.cousin_conv1(self.relu(cousin_out))
        # print(cousin_out.shape)
        cousin_out = self.cousin_conv2(self.relu(cousin_out))
        cousin_out = self.cousin_conv3(self.relu(cousin_out))
        cousin_out = self.cousin_conv4(self.relu(cousin_out))
        cousin_out = self.cousin_conv5(self.relu(cousin_out))
        cousin_out = self.cousin_conv6(self.relu(cousin_out))
        cousin_out = torch.add(cousin_out, cousin_res)
        """
        ### visualization
        a = out.squeeze(0)
        b = cousin_out.squeeze(0)
        for i in range (0,128,16):
            alpha = a[i].cpu().data.detach().numpy()
            alpha = ((alpha - alpha.min()) / (alpha.max() - alpha.min()) )
            beta = b[i].cpu().data.detach().numpy()
            beta = ((beta - beta.min()) / (beta.max()-beta.min()))
            plt.imshow(beta, cmap='gray')
            plt.show()
            #plt.imshow(alpha)
            #plt.show()
        ###
        """
        out = torch.add(out, cousin_out)

        out = self.output(self.relu(out))

        out = torch.add(out, residual)

        return out

class simpleNet(nn.Module):
    def __init__(self, Y=True):
        super(simpleNet, self).__init__()
        d = 1
        if Y == False:
            d = 3
        self.input = nn.Conv2d(in_channels=d, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)

        self.output = nn.Conv2d(in_channels=128, out_channels=d, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = x
        inputs = self.input(self.relu(x))
        out = inputs

        out = self.conv1(self.relu(out))
        out = self.conv2(self.relu(out))
        out = self.conv3(self.relu(out))
        out = self.conv4(self.relu(out))
        out = self.conv5(self.relu(out))
        out = self.conv6(self.relu(out))

        # out = torch.add(out, inputs)

        out = self.output(self.relu(out))

        out = torch.add(out, residual)
        return out


class _NonLocalBlockND_v2(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND_v2, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, y):  # y : reference , x : LR_up
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_y = self.g(y).view(batch_size, self.inter_channels, -1)
        g_y = g_y.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_y = self.phi(y).view(batch_size, self.inter_channels, -1)


        f = torch.matmul(theta_x, phi_y)

        # print(f[0][0])
        # print(f)
        f_div_C = F.softmax(f, dim=-1)
        # print(f_div_C)
        k = torch.matmul(f_div_C, g_y)
        k = k.permute(0, 2, 1).contiguous()
        k = k.view(batch_size, self.inter_channels, *x.size()[2:])
        W_k = self.W(k)
        z = W_k + x

        return z

"""
class _NonLocalBlockND_v2(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND_v2, self).__init__()
        assert dimension in [1, 2, 3]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z
"""

class NONLocalBlock2D(_NonLocalBlockND_v2):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class RZSR(nn.Module):
    def __init__(self, Y=True):
        super(RZSR, self).__init__()
        d = 1
        if Y == False:
            d = 3
        self.input = nn.Conv2d(in_channels=d, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.ResBlcok1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.ResBlcok2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.ResBlcok3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.ResBlcok4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.Downscalex2_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.Downscalex4_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.Downscalex8_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.Nonlocal_block = NONLocalBlock2D(in_channels=128, inter_channels=64, sub_sample=False, bn_layer=False)

        self.Upscalex8_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.Upscalex4_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.Upscalex2_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.output = nn.Conv2d(in_channels=128, out_channels=d, kernel_size=3, stride=1, padding=1, bias=False)

        self.Reference_input = nn.Conv2d(in_channels=d, out_channels=128, kernel_size=3, stride=1, padding=1,
                                         bias=False)
        self.Ref_ResBlcok1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.Ref_ResBlcok2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.Ref_ResBlcok3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.Ref_ResBlcok4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.Ref_Downscalex2_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.Ref_Downscalex4_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.Ref_Downscalex8_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.LastConv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x, y): #
        input = x
        x = self.input(x)
        x = self.ResBlcok1(x)
        x = self.ResBlcok2(x)
        x = self.ResBlcok3(x)
        x = self.ResBlcok4(x)
        upper_level1 = x

        y = self.Reference_input(y)
        y = self.ResBlcok1(y)
        y = self.ResBlcok2(y)
        y = self.ResBlcok3(y)
        y = self.ResBlcok4(y)
        level_1 = y
        level_2 = self.Downscalex2_conv(y)
        level_4 = self.Downscalex4_conv(level_2)
        level_8 = self.Downscalex8_conv(level_4)

        upper_level_2 = self.Downscalex2_conv(x)
        upper_level_4 = self.Downscalex4_conv(upper_level_2)
        upper_level_8 = self.Downscalex8_conv(upper_level_4)

        non_x = self.Nonlocal_block(upper_level_8, level_8) #+ upper_level_8
        non_x = self.Upscalex8_conv(non_x)
        non_x = self.Nonlocal_block(non_x, level_4) + upper_level_4
        non_x = self.Upscalex4_conv(non_x)
        non_x = self.Nonlocal_block(non_x, level_2) + upper_level_2
        non_x = self.Upscalex2_conv(non_x)

        non_x = self.LastConv(non_x) #+ level_1

        out = self.output(non_x)

        out = torch.add(out, input)

        return out


class RZSR_no_ref(nn.Module):
    def __init__(self, Y=True):
        super(RZSR_no_ref, self).__init__()
        d = 1
        if Y == False:
            d = 3
        self.input = nn.Conv2d(in_channels=d, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.ResBlcok1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.ResBlcok2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.ResBlcok3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.ResBlcok4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.Downscalex2_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.Downscalex4_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.Downscalex8_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.Nonlocal_block1 = NONLocalBlock2D(in_channels=128, inter_channels=128, sub_sample=False, bn_layer=False)
        self.Nonlocal_block2 = NONLocalBlock2D(in_channels=128, inter_channels=128, sub_sample=False, bn_layer=False)
        self.Nonlocal_block3 = NONLocalBlock2D(in_channels=128, inter_channels=128, sub_sample=False, bn_layer=False)

        self.Upscalex8_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.Upscalex4_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.Upscalex2_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.output = nn.Conv2d(in_channels=128, out_channels=d, kernel_size=3, stride=1, padding=1, bias=False)

        self.Reference_input = nn.Conv2d(in_channels=d, out_channels=128, kernel_size=3, stride=1, padding=1,
                                         bias=False)
        self.Ref_ResBlcok1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.Ref_ResBlcok2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.Ref_ResBlcok3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.Ref_ResBlcok4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.Ref_Downscalex2_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.Ref_Downscalex4_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.Ref_Downscalex8_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.LastConv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x): ################### No-reference Module -- Please Change _NonLocalBlockND_v2 
        input = x
        x = self.input(x)
        x = self.ResBlcok1(x)
        x = self.ResBlcok2(x)
        x = self.ResBlcok3(x)
        x = self.ResBlcok4(x)
        upper_level1 = x

        upper_level_2 = self.Downscalex2_conv(x)
        upper_level_4 = self.Downscalex4_conv(upper_level_2)
        upper_level_8 = self.Downscalex8_conv(upper_level_4)

        non_x = self.Nonlocal_block1(upper_level_8) #+ upper_level_8
        non_x = self.Upscalex8_conv(non_x)
        non_x = self.Nonlocal_block2(non_x) + upper_level_4
        non_x = self.Upscalex4_conv(non_x)
        non_x = self.Nonlocal_block3(non_x) + upper_level_2
        non_x = self.Upscalex2_conv(non_x)

        non_x = self.LastConv(non_x) #+ level_1

        out = self.output(non_x)

        out = torch.add(out, input)

        return out

"""
class RZSR_single_scale(nn.Module):
    def __init__(self, Y=True):
        super(RZSR_single_scale, self).__init__()
        d = 1
        if Y == False:
            d = 3
        self.input = nn.Conv2d(in_channels=d, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.ResBlcok1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.ResBlcok2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.ResBlcok3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.ResBlcok4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.Nonlocal_block = NONLocalBlock2D(in_channels=128, inter_channels=64, sub_sample=False, bn_layer=False)


        self.output = nn.Conv2d(in_channels=128, out_channels=d, kernel_size=3, stride=1, padding=1, bias=False)

        self.Reference_input = nn.Conv2d(in_channels=d, out_channels=128, kernel_size=3, stride=1, padding=1,
                                         bias=False)
        self.Ref_ResBlcok1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.Ref_ResBlcok2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.Ref_ResBlcok3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.Ref_ResBlcok4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )


        self.LastConv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x, y): ###################################3 Non local 바꾸기만 해주면 될
        input = x
        x = self.input(x)
        x = self.ResBlcok1(x)
        x = self.ResBlcok2(x)
        x = self.ResBlcok3(x)
        x = self.ResBlcok4(x)

        y = self.Reference_input(y)
        y = self.Ref_ResBlcok1(y)
        y = self.Ref_ResBlcok2(y)
        y = self.Ref_ResBlcok3(y)
        y = self.Ref_ResBlcok4(y)


        non_x = self.Nonlocal_block(x,y)
        non_x = self.LastConv(non_x) #+ level_1

        out = self.output(non_x)

        out = torch.add(out, input)

        return out


    """
