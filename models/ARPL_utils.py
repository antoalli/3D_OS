from torch import nn
import torch.nn.functional as F


class _MultiBatchNorm(nn.Module):
    """Taken from https://github.com/iCGY96/ARPL
    Modified to support BN with different dimensions

    """
    _version = 2

    def __init__(self, num_features, num_classes, bn_dims=2, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_MultiBatchNorm, self).__init__()
        self.bn_dims = bn_dims
        if bn_dims == 2:
            self.bns = nn.ModuleList(
                [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])
        elif bn_dims == 1:
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])
        else:
            raise NotImplementedError(f"Number of BN dims not supported {bn_dims}")

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        bn = self.bns[domain_label]
        return bn(x)


class MultiBatchNorm(_MultiBatchNorm):
    def _check_input_dim(self, input, expected_dim=4):
        if self.bn_dims == 1:
            if input.dim() not in [2, 3]:
                raise ValueError(f'expected 3D or 2D input (got {input.dim()}D input)')
            return
        if input.dim() != 4:
            raise ValueError(f'expected 4D input (got {input.dim()}D input)')


class Discriminator(nn.Module):
    def __init__(self, features=[3, 64, 128, 256, 512, 1024]):
        self.layer_num = len(features) - 1
        super().__init__()

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(features[inx], features[inx + 1], kernel_size=1, stride=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.final_layer = nn.Sequential(
            nn.Linear(features[-1], features[-1]),
            nn.Linear(features[-1], features[-2]),
            nn.Linear(features[-2], features[-2]),
            nn.Linear(features[-2], 1),
            nn.Sigmoid())

    def forward(self, f):
        feat = f.transpose(1, 2)
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)

        out = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        out = self.final_layer(out)  # (B, 1)
        return out


class Generator(nn.Module):
    def __init__(self, z_dim=96, num_points=2048):
        super().__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, self.num_points * 3)
        self.th = nn.Tanh()

    def forward(self, x):
        bs = x.size()[0]
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        x = self.th(self.fc4(x))
        x = x.view(bs, 3, self.num_points)
        return x.transpose(2, 1)

