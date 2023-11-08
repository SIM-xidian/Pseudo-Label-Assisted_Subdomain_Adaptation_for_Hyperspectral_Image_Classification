import torch.nn as nn
import torch
class EmbeddingNetHyperX(nn.Module):
    def __init__(self, input_channels, n_outputs=128, patch_size=5, n_classes=None):
        super(EmbeddingNetHyperX, self).__init__()
        self.dim = 200

        # 1st conv layer
        # input [input_channels x patch_size x patch_size]
        self.convnet = nn.Sequential(
            nn.Conv2d(input_channels, self.dim, kernel_size=1, padding=0),  # input channels
            nn.BatchNorm2d(self.dim, momentum=1, affine=True,track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.dim, momentum=1, affine=True,track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.dim, momentum=1, affine=True,track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=1, padding=0,),
            nn.BatchNorm2d(self.dim, momentum=1, affine=True,track_running_stats=True),
            nn.ReLU(),

            nn.AvgPool2d(patch_size, stride=1)

        )

        self.n_outputs = n_outputs
        self.fc = nn.Linear(self.dim, self.n_outputs)

    def extract_features(self, x):

        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc[0](output)

        return output

    def forward(self, x):

        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)

        return output

    def get_embedding(self, x):
        return self.forward(x)
    def output_num(self):
        return self.n_outputs

class ResClassifier(nn.Module):
    def __init__(self, num_classes=7,  num_unit=128, middle=64):
        super(ResClassifier, self).__init__()
        layers = []

        layers.append(nn.Linear(num_unit, middle))
        layers.append(nn.BatchNorm1d(middle, affine=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(middle, middle))
        layers.append(nn.BatchNorm1d(middle, affine=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(middle, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):

        x = self.classifier(x)
        return x

class SPM_CAM(nn.Module):
    def __init__(self):
        super(SPM_CAM, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(5, 2), stride=(1, 1), padding=(2, 0), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, eps=1e-5):
        # feature descriptor on the global spatial information
        N, C, height, width = x.size()
        channel_center = x.view(N, C, -1)[:, :, int((x.shape[2] * x.shape[3] - 1) / 2)]
        channel_center = channel_center.unsqueeze(2)
        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_center), dim=2)
        spp = self.conv(t.unsqueeze(1)).transpose(1, 2)
        y = self.sigmoid(spp)

        return x * (1 + y.expand_as(x))


class SPAN(nn.Module):
    def __init__(self, band, classes):
        super(SPAN, self).__init__()
        # dimension reduction
        self.n_outputs = 128
        self.conv_DR = nn.Sequential(
                # LKA(band, 30),
                nn.Conv2d(in_channels=band, out_channels=30, kernel_size=1, padding=0, stride=1),
                nn.BatchNorm2d(30, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
                nn.ReLU(inplace=True)
            )

        # 光谱先验空间自注意力
        self.SPM_CAM = SPM_CAM()

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, padding=(0, 0, 1),
                      kernel_size=(1, 1, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(32, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, padding=(0, 0, 1),
                      kernel_size=(1, 1, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(32, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, padding=(0, 0, 1),
                      kernel_size=(1, 1, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(32, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.conv3d_4 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=90, padding=(0, 0, 0),
                      kernel_size=(1, 1, 30), stride=(1, 1, 1)),
            nn.BatchNorm3d(90, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(30, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(30, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.conv2d_3 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(30, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.convC = nn.Sequential(
            nn.Conv2d(in_channels=120, out_channels=120, padding=0, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(120, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(120, self.n_outputs)


    def get_embedding(self, x):
        return self.forward(x)
    
    def output_num(self):
        return self.n_outputs

    def forward(self, X):
        X = self.SPM_CAM(X)  # X.shape： # [bz, 204, 9, 9]
        X = self.conv_DR(X)
        spe1 = self.conv3d_1(X.permute(0, 2, 3, 1).unsqueeze(1))
        spe2 = self.conv3d_2(spe1)

        spa1 = self.conv2d_1(X)
        spa2 = self.conv2d_2(spa1)
        spa3 = self.conv2d_3(spa2)

        spe3 = self.conv3d_3(spe2)
        spe4 = self.conv3d_4(spe3)

        spe4 = spe4.squeeze(-1)

        ss = torch.cat((spa3, spe4), dim=1)
        ss = self.convC(ss)
        ss = self.gap(ss)
        ss = ss.view(ss.size(0), -1)
        output = self.fc(ss)
        return output
