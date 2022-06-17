import torch
import torch.nn as nn


class PointNetFeat(nn.Module):
    """
    Simple PointNet encoder without spatial and feature transformer
    """

    def __init__(self, dim_in=3, **kwargs):
        super(PointNetFeat, self).__init__()

        self.dim_in = dim_in

        self.conv1 = nn.Sequential(
            nn.Conv1d(dim_in, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 1024, 1, bias=False),
            nn.BatchNorm1d(1024)
        )

    def forward(self, x):
        bs, dim1, dim2 = x.size()
        if dim2 == self.dim_in:
            # bnc -> bcn
            x = x.transpose(2, 1)

        npoints = x.size(-1)

        x = self.conv1(x)
        x = self.conv2(x)
        x_skip = self.conv3(x)  # [bs,64,npoints]
        x = self.conv4(x_skip)
        x = self.conv5(x)

        x_class = torch.max(x, 2, keepdim=True)[0]
        x_class = x_class.view(-1, 1024)  # [bs, 1024]

        x_segm = x_class.view(-1, 1024, 1).repeat(1, 1, npoints)
        x_segm = torch.cat([x_segm, x_skip], 1)  # [bs, 1088, npoints]

        return x_class, x_segm
