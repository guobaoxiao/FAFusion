# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

class DenseBlock(nn.Module):
    def __init__(self,channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)
    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x

class RGBD(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(RGBD, self).__init__()
        self.dense =DenseBlock(in_channels)
        self.convdown=Conv1(3*in_channels,out_channels)
        self.sobelconv=Sobelxy(in_channels)
        self.convup =Conv1(in_channels,out_channels)
    def forward(self,x):
        x1=self.dense(x)
        x1=self.convdown(x1)
        x2=self.sobelconv(x)
        x2=self.convup(x2)
        return F.leaky_relu(x1+x2,negative_slope=0.1)

class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False, use_cbam=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        # elif norm == 'adain':
        #     self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if norm == 'sn':
            self.conv = nn.utils.spectral_norm(nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

        if use_cbam:
            self.cbam = CBAM(out_dim, 16, no_spatial=True)
        else:
            self.cbam = None

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.cbam:
                x = self.cbam(x)
            if self.activation:
                x = self.activation(x)
        return x
        
class EB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(EB, self).__init__()
        self.forw = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return F.leaky_relu(self.forw(x), negative_slope=0.2)

class DEB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DEB, self).__init__()
        self.forw = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0,bias=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return F.leaky_relu(self.forw(x), negative_slope=0.2)
        
class Gradnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Gradnet, self).__init__()
        self.en1 = EB(in_channels, in_channels*2)
        self.en2 = EB(in_channels*2, in_channels*3)
        self.en3 = EB(in_channels*3, in_channels*4)
        self.en4 = EB(in_channels*4, in_channels*5)
        self.sobel1 = Sobelxy(in_channels*2)
        self.sobel2 = Sobelxy(in_channels*3)
        self.sobel3 = Sobelxy(in_channels*4)
        self.de1 = DEB(in_channels*5, in_channels*4)
        self.de2 = DEB(in_channels*4, in_channels*3)
        self.de3 = DEB(in_channels*3, in_channels*2)
        self.de4 = DEB(in_channels*2, in_channels)
        
    def forward(self, x):
        x1 = self.en1(x)
        x2 = self.en2(x1)
        x3 = self.en3(x2) 
        x4 = self.en4(x3)
          
        x5 = self.sobel1(x1)
        x6 = self.sobel2(x2)
        x7 = self.sobel3(x3)
        
        x8 = self.de1(x4)
        x9 = self.de2(x7+x8)
        x10 = self.de3(x9+x6)
        x11 = self.de4(x10+x5)
        
        return x11

def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
   #  print(filter_LL.size())
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)

    return LL, LH, HL, HH

class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

def get_wav_two(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
   #  print(filter_LL.size())
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH


    
class WavePool2(nn.Module):
    def __init__(self, in_channels):
        super(WavePool2, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav_two(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav_two(self.in_channels, pool=False)

    def forward(self,LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return  self.LH(LH) + self.HL(HL) + self.HH(HH)  #进行反卷积
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        elif self.option_unpool =='sumall':
            return   LH + HL + HH     #直接相加，不进行反卷积
        else:
            raise NotImplementedError


                                               
class FE(nn.Module):
    def __init__(self,
                 in_channels, out_channels1, out_channels2, out_channels3, out_channels4,
                 ksize=3, stride=1, pad=1):

        super(FE, self).__init__()
        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels1, 1, 1, 0),
            #nn.BatchNorm2d(out_channels1),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels2, 1, 1, 0),
            #nn.BatchNorm2d(out_channels2),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.body3 = nn.Sequential(
            nn.Conv2d(out_channels2, out_channels2,  ksize, stride, pad),
            #nn.BatchNorm2d(out_channels2),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.body4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels3, 1, 1, 0),
            #nn.BatchNorm2d(out_channels3),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.body5 = nn.Sequential(
            nn.Conv2d(out_channels2, out_channels3, ksize, stride, pad),
            #nn.BatchNorm2d(out_channels3),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(out_channels2, out_channels3, ksize, stride, pad),
            #nn.BatchNorm2d(out_channels3),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.body6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels4, 1, 1, 0),
            #nn.BatchNorm2d(out_channels4),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.body7 = nn.Sequential(
            nn.Conv2d(out_channels4, out_channels4, ksize, stride, pad),
            #nn.BatchNorm2d(out_channels4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(out_channels4, out_channels4, ksize, stride, pad),
            #nn.BatchNorm2d(out_channels4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(out_channels4, out_channels4, ksize, stride, pad),
            #nn.BatchNorm2d(out_channels4),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        out1 = self.body1(x)

        out2 =  self.body3(self.body2(x)+out1)

        out3 = self.body5(self.body4(x)+out2)

        out4 = self.body7(self.body6(x)+out3)

        out = torch.cat([out1, out2, out3, out4], dim=1)

        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.Upsample = nn.Upsample(scale_factor=2)
        
        self.conv1 = Conv2dBlock(64, 32, 5, 1, 2,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.sobel1 = Sobelxy(32)
        self.pool1 = WavePool2(32).cuda()
        self.conv6 = Conv2dBlock(32, 64, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        
        self.conv2 = Conv2dBlock(32, 64, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.sobel2 = Sobelxy(64)
        self.pool2 = WavePool2(64).cuda()
        self.conv7 = Conv2dBlock(64, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')

        
        self.conv3 = Conv2dBlock(64, 128, 3,1 , 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.sobel3 = Sobelxy(128)
        self.pool3 = WavePool2(128).cuda()
        self.conv8 = Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')

        
        self.conv4 = Conv2dBlock(128, 128, 3,1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.sobel4 = Sobelxy(128)
        self.pool4 = WavePool2(128).cuda()
        self.conv9 = Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')

        
        self.conv5 = Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
    def forward(self, x): 
        #(24,3,128,128)
        skips = {}
        x = self.conv1(x)
        x_sobel1 = self.sobel1(x)
        #(24,32,128,128)
        skips['conv1_1'] = x
        LL1, LH1, HL1, HH1 = self.pool1(x)
        LL1_up = self.Upsample(LL1)
        LL1_up_conv = self.conv6(LL1_up)
        
        # (24,64,64,64)
        skips['pool1'] = [LH1, HL1, HH1]
        x = self.conv2(x)
        x_sobel2 = self.sobel2(x)
        #(24,64,64,64)`
        # p2 = self.pool2(x)
        #（24,128,32,32）
        skips['conv2_1'] = x
        LL2, LH2, HL2, HH2 = self.pool2(x)
        LL2_up = self.Upsample(LL2)
        LL2_up_conv = self.conv7(LL2_up)
        #（24,128,32,32）
        skips['pool2'] = [LH2, HL2, HH2]

        x = self.conv3(x+LL1_up_conv)
        x_sobel3 = self.sobel3(x)
        #(24,128,32,32)
        # p3 = self.pool3(x)
        skips['conv3_1'] = x
        LL3, LH3, HL3, HH3 = self.pool3(x)
        LL3_up = self.Upsample(LL3)
        LL3_up_conv = self.conv8(LL3_up)
        #(24,128,16,16)
        skips['pool3'] = [LH3, HL3, HH3]
        #(24,128,32,32)
        x = self.conv4(x+LL2_up_conv)
        x_sobel4 = self.sobel4(x)

        #(24,128,16,16)
        skips['conv4_1'] = x
        LL4, LH4, HL4, HH4 = self.pool4(x)
        LL4_up = self.Upsample(LL4)
        LL4_up_conv = self.conv9(LL4_up)
        skips['pool4'] = [LH4, HL4, HH4]
        #(24,128,8,8)
        x = self.conv5(x+LL3_up_conv)
        x = x + LL4_up_conv
        #(24,128,8,8)
        return x, skips, x_sobel4,x_sobel3,x_sobel2,x_sobel1

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.Upsample = nn.Upsample(scale_factor=2)
        self.Conv1 = Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block1 = WaveUnpool(128,"sum").cuda()  #进行反卷积
        self.Conv2 = Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block2 = WaveUnpool(128, "sum").cuda()
        self.Conv3 = Conv2dBlock(128, 64, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block3 = WaveUnpool(64, "sum").cuda()  # 直接相加，不进行反卷积
        self.Conv4 = Conv2dBlock(64, 32, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.recon_block4 = WaveUnpool(32, "sum").cuda()

        self.Conv6 = Conv2dBlock(32, 1, 3, 1, 1,
                             norm='bn',
                             activation='tanh',
                             pad_type='reflect')
    def forward(self, x, skips,x_sobel4,x_sobel3,x_sobel2,x_sobel1):
        #x1 = self.Upsample(x)
        x2 = self.Conv1(x)
        LH1, HL1, HH1 = skips['pool4']
        original1 = skips['conv4_1']
        x_deconv = self.recon_block1(LH1, HL1, HH1, original1)
        LH2, HL2, HH2 = skips['pool3']
        original2 = skips['conv3_1']
        x_deconv2 = self.recon_block2(LH2, HL2, HH2, original2)
        x2 = x_deconv + x2 +x_sobel4
        
        #x3 = self.Upsample(x2)
        x4 = self.Conv2(x2)
        LH3, HL3, HH3 = skips['pool2']
        original3 = skips['conv2_1']
        x_deconv3 = self.recon_block3(LH3, HL3, HH3, original3)
        x4 = x_deconv2 + x4 +x_sobel3
        
        #LH3, HL3, HH3 = skips['pool2']
        #original3 = skips['conv2_1']
        #x_deconv3 = self.recon_block1(LH3, HL3, HH3, original2)
        #x5 = self.Upsample(x4+x_deconv2)
        x6 = self.Conv3(x4)
        LH4, HL4, HH4 = skips['pool1']
        original4 = skips['conv1_1']
        x_deconv4 = self.recon_block4(LH4, HL4, HH4, original4)

        #x7 = self.Upsample(x6+x_deconv4)
        #x7 = self.Upsample(x6)
        x8 = self.Conv4(x6+x_deconv3+x_sobel2)
        
        x10=self.Conv6(x8+x_deconv4+x_sobel1)
        return x10     
            
class Wavenet(nn.Module):
    def __init__(self):
        super(Wavenet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        fea,skips, x_sobel4,x_sobel3,x_sobel2,x_sobel1 = self.encoder(x)
        Fuse = self.decoder(fea, skips, x_sobel4,x_sobel3,x_sobel2,x_sobel1)
        
        return Fuse

class FusionNet(nn.Module):
    def __init__(self, output):
        super(FusionNet, self).__init__()
        self.fe = FE(2, 16, 16, 16, 16)#64
        #self.grad = Gradnet(64,64)
        self.wave = Wavenet()
        
        
    def forward(self, image_vis,image_ir):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        #x_vis_origin = image_vis
        x_inf_origin = image_ir
        input = torch.cat([x_vis_origin, x_inf_origin], dim=1)
        # encode
        features = self.fe(input)
        #grad = self.grad(features)
        wave_feature = self.wave(features)
        
        return wave_feature

def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(2,4,480,640).astype(np.float32))
    model = FusionNet(output=1)
    y = model(x)
    print('output shape:', y.shape)
    assert y.shape == (2,1,480,640), 'output shape (2,1,480,640) is expected!'
    print('test ok!')

if __name__ == '__main__':
    unit_test()
