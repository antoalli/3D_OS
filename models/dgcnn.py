import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ARPL_utils import MultiBatchNorm


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class DGCNN(nn.Module):
    def __init__(self, k=20, emb_dims=1024, **kwargs):
        super(DGCNN, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        print(f"dgcnn k: {self.k}")
        print(f"dgcnn emb_dims: {self.emb_dims}")

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2))

        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(
            nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.emb_dims),
            nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        batch_size, num_points, num_dims = x.size()
        assert num_dims == 3, "expected BNC shape as input"
        x = x.transpose(2, 1)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        res = torch.cat((x1, x2), 1)
        return res


class DGCNNABN(nn.Module):
    """
    DGCNN encoder with Auxiliary Batch Norm from ARPL (https://github.com/iCGY96/ARPL)

    We will pass through this network both real and fake data and we want to have different
    BN stats for the two distributions
    """

    def __init__(self, k=20, emb_dims=1024, **kwargs):
        super().__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.num_ABN = 2
        print(f"DGCNNABN k: {self.k}")
        print(f"DGCNNABN emb_dims: {self.emb_dims}")
        print(f"DGCNNABN num_ABN: {self.num_ABN}")

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = MultiBatchNorm(64, num_classes=self.num_ABN, bn_dims=2)

        self.conv2 = nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(64)
        self.bn2 = MultiBatchNorm(64, num_classes=self.num_ABN, bn_dims=2)

        self.conv3 = nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(128)
        self.bn3 = MultiBatchNorm(128, num_classes=self.num_ABN, bn_dims=2)

        self.conv4 = nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False)
        # self.bn4 = nn.BatchNorm2d(256)
        self.bn4 = MultiBatchNorm(256, num_classes=self.num_ABN, bn_dims=2)

        self.conv5 = nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False)
        # self.bn5 = nn.BatchNorm1d(self.emb_dims)
        self.bn5 = MultiBatchNorm(self.emb_dims, num_classes=self.num_ABN, bn_dims=1)

    def forward(self, x, bn_label=0):
        batch_size, num_points, num_dims = x.size()
        assert num_dims == 3, "expected BNC shape as input"
        x = x.transpose(2, 1)

        x = get_graph_feature(x, k=self.k)

        # conv1
        x = self.conv1(x)
        x = self.bn1(x, bn_label)
        x = self.leaky_relu(x)

        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        # conv2
        x = self.conv2(x)
        x = self.bn2(x, bn_label)
        x = self.leaky_relu(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        # conv3
        x = self.conv3(x)
        x = self.bn3(x, bn_label)
        x = self.leaky_relu(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        # conv4
        x = self.conv4(x)
        x = self.bn4(x, bn_label)
        x = self.leaky_relu(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        # conv5
        x = self.conv5(x)
        x = self.bn5(x, bn_label)
        x = self.leaky_relu(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        res = torch.cat((x1, x2), 1)
        return res


if __name__ == '__main__':
    data = torch.rand(32, 1024, 3).cuda()
    print("Input: ", data.shape)
    print("===> testing DGCNN ...")
    model = DGCNNABN().cuda()
    out = model(data)
    print("Output: ", out.shape)  # [bs, 2048]
