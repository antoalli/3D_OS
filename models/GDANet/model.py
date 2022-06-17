import torch.nn as nn
import torch
import torch.nn.functional as F
from models.GDANet.GDANet_util import local_operator, GDM, SGCAM


class GDANET(nn.Module):
    def __init__(self, ):
        super(GDANET, self).__init__()
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn11 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn12 = nn.BatchNorm1d(64, momentum=0.1)

        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn21 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn22 = nn.BatchNorm1d(64, momentum=0.1)

        self.bn3 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn31 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn32 = nn.BatchNorm1d(128, momentum=0.1)

        self.bn4 = nn.BatchNorm1d(512, momentum=0.1)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=True),
                                   self.bn1)
        self.conv11 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                    self.bn11)
        self.conv12 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=True),
                                    self.bn12)

        self.conv2 = nn.Sequential(nn.Conv2d(67 * 2, 64, kernel_size=1, bias=True),
                                   self.bn2)
        self.conv21 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                    self.bn21)
        self.conv22 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=True),
                                    self.bn22)

        self.conv3 = nn.Sequential(nn.Conv2d(131 * 2, 128, kernel_size=1, bias=True),
                                   self.bn3)
        self.conv31 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=True),
                                    self.bn31)
        self.conv32 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=True),
                                    self.bn32)

        self.conv4 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, bias=True),
                                   self.bn4)

        self.SGCAM_1s = SGCAM(64)
        self.SGCAM_1g = SGCAM(64)
        self.SGCAM_2s = SGCAM(64)
        self.SGCAM_2g = SGCAM(64)

    def forward(self, x):
        batch_size, num_points, num_dims = x.size()
        assert num_dims == 3, "expected BNC shape as input"
        x = x.permute(0, 2, 1)

        ###############
        """block 1"""
        # Local operator:
        x1 = local_operator(x, k=30)
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv11(x1))
        x1 = x1.max(dim=-1, keepdim=False)[0]

        # Geometry-Disentangle Module:
        x1s, x1g = GDM(x1, M=256)

        # Sharp-Gentle Complementary Attention Module:
        y1s = self.SGCAM_1s(x1, x1s.transpose(2, 1))
        y1g = self.SGCAM_1g(x1, x1g.transpose(2, 1))
        z1 = torch.cat([y1s, y1g], 1)
        z1 = F.relu(self.conv12(z1))
        ###############
        """block 2"""
        x1t = torch.cat((x, z1), dim=1)
        x2 = local_operator(x1t, k=30)
        x2 = F.relu(self.conv2(x2))
        x2 = F.relu(self.conv21(x2))
        x2 = x2.max(dim=-1, keepdim=False)[0]

        x2s, x2g = GDM(x2, M=256)

        y2s = self.SGCAM_2s(x2, x2s.transpose(2, 1))
        y2g = self.SGCAM_2g(x2, x2g.transpose(2, 1))
        z2 = torch.cat([y2s, y2g], 1)
        z2 = F.relu(self.conv22(z2))
        ###############
        x2t = torch.cat((x1t, z2), dim=1)
        x3 = local_operator(x2t, k=30)
        x3 = F.relu(self.conv3(x3))
        x3 = F.relu(self.conv31(x3))
        x3 = x3.max(dim=-1, keepdim=False)[0]
        z3 = F.relu(self.conv32(x3))
        ###############
        x = torch.cat((z1, z2, z3), dim=1)
        x = F.relu(self.conv4(x))
        x11 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x22 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x11, x22), 1)  # [bs, 1024]
        return x


if __name__ == '__main__':
    data = torch.rand(32, 1024, 3).cuda()
    print("Input: ", data.shape)
    print("===> testing GDANET ...")
    model = GDANET().cuda()
    out = model(data)
    print("Output: ", out.shape)  # [32, 1024]


