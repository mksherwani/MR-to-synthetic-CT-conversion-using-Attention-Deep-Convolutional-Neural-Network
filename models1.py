import torch.nn as nn
import torch


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class Net1(nn.Module):
    def __init__(self, lp):
        super(Net1, self).__init__()

        initial_features = lp["filtersInitNum"]
        num_channels = len(lp["MRIchannels"])
        dropout = lp["dropout"]

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)


        self.conv_down1 = SingleConv3x3(num_channels, initial_features, dropout)
        self.conv_down2 = SingleConv3x3(initial_features, initial_features, dropout)
        self.conv_down3 = DoubleConv3x3(initial_features, initial_features*2, dropout)
        self.conv_down4 = TripleConv3x3(initial_features*2, initial_features*(2**2), dropout)
        self.conv_down5 = TripleConv3x3(initial_features*(2**2), initial_features*(2**3), dropout)
        self.conv_down6 = TripleConv3x3(initial_features*(2**3), initial_features*(2**3), dropout)

        self.conv_down7 = TripleConv3x3(initial_features*(2**3), initial_features*(2**3), dropout)



        self.upconcat1 = UpConcat(initial_features*(2**3), initial_features*(2**3))
        self.conv_up1_2_3 = TripleConv3x3(initial_features*(2**3)+initial_features*(2**3), initial_features*(2**2), dropout)


        self.upconcat2 = UpConcat(initial_features*(2**3), initial_features*(2**2))
        self.conv_up4_5_6 = TripleConv3x3(initial_features*(2**2)+initial_features*(2**2), initial_features*2, dropout)


        self.upconcat3 = UpConcat(initial_features*(2**2), initial_features*2)
        self.conv_up7_8 = DoubleConv3x3(initial_features*2+initial_features*2, initial_features, dropout)


        self.upconcat4 = UpConcat(initial_features*2, initial_features)
        self.Att = Attention_block(F_g=64, F_l=64, F_int=32)
        self.conv_up9_10 = DoubleConv3x3(initial_features+initial_features, initial_features, dropout)
        

        self.final = SingleConv1x1(initial_features, dropout)

    def forward(self, inputs):

        conv_down1_feat = self.conv_down1(inputs)
        conv_down2_feat = self.conv_down2(conv_down1_feat) ####
        maxpool1_feat = self.maxpool1(conv_down2_feat)

        conv_down3_feat = self.conv_down3(maxpool1_feat) ###
        maxpool2_feat = self.maxpool2(conv_down3_feat)

        conv_down4_feat = self.conv_down4(maxpool2_feat) ##
        maxpool3_feat = self.maxpool3(conv_down4_feat)

        conv_down5_feat = self.conv_down5(maxpool3_feat) #
        maxpool4_feat = self.maxpool4(conv_down5_feat)

        conv_down6_feat = self.conv_down6(maxpool4_feat)
        conv_down7_feat = self.conv_down7(conv_down6_feat)

        upconcat1_feat = self.upconcat1(conv_down7_feat, conv_down5_feat)
        conv_up1_2_3_feat = self.conv_up1_2_3(upconcat1_feat)

        upconcat2_feat = self.upconcat2(conv_up1_2_3_feat, conv_down4_feat)
        conv_up4_5_6_feat = self.conv_up4_5_6(upconcat2_feat)

        upconcat3_feat = self.upconcat3(conv_up4_5_6_feat, conv_down3_feat)
        conv_up7_8_feat = self.conv_up7_8(upconcat3_feat)

        upconcat4_feat = self.upconcat4(conv_up7_8_feat, conv_down2_feat)
        x1 = self.Att(g=upconcat4_feat, x=conv_down2_feat)
        upconcat4_feat = torch.cat((x1, upconcat4_feat))
        conv_up9_10_feat = self.conv_up9_10(upconcat4_feat)

        outputs = self.final(conv_up9_10_feat)

        return outputs

class SingleConv1x1(nn.Module):
    def __init__(self, in_feat, dropout):
        super(SingleConv1x1, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, 1,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0),
                                   nn.Dropout2d(dropout))

    def forward(self, inputs):
        outputs = self.conv1(inputs)

        return outputs


class SingleConv3x3(nn.Module):
    def __init__(self, in_feat, out_feat, dropout):
        super(SingleConv3x3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.Dropout2d(dropout),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class DoubleConv3x3(nn.Module):
    def __init__(self, in_feat, out_feat, dropout):
        super(DoubleConv3x3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, in_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.Dropout2d(dropout),
                                   nn.BatchNorm2d(in_feat),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.Dropout2d(dropout),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class TripleConv3x3(nn.Module):
    def __init__(self, in_feat, out_feat, dropout):
        super(TripleConv3x3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, in_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.Dropout2d(dropout),
                                   nn.BatchNorm2d(in_feat),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(in_feat, in_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.Dropout2d(dropout),
                                   nn.BatchNorm2d(in_feat),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.Dropout2d(dropout),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs

class UpConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()

        self.deconv = nn.ConvTranspose2d(out_feat, out_feat, kernel_size=3,
                                         padding=1, stride=1, dilation=1, output_padding=0)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        
    def forward(self, inputs, down_outputs):
        outputs = self.deconv(inputs)
        outputs = self.up(outputs)
        out = torch.cat([down_outputs, outputs], 1)
        return out
