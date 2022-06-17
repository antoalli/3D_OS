"""
@Author: Tiange Xiang
@Contact: txia7609@uni.sydney.edu.au
@File: curvenet_cls.py
@Time: 2021/01/21 3:10 PM
"""
from models.CurveNet.curvenet_util import *

curve_config = {
    'default': [[100, 5], [100, 5], None, None],
    'long': [[10, 30], None, None, None]
}


class CurveNet(nn.Module):
    def __init__(self, k=20, setting='default'):
        super(CurveNet, self).__init__()
        assert setting in curve_config
        print(f"CurveNet k: {k}")
        print(f"CurveNet setting: {setting}")

        additional_channel = 32
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # encoder
        self.cic11 = CIC(npoint=1024, radius=0.05, k=k, in_channels=additional_channel, output_channels=64,
                         bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][0])
        self.cic12 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4,
                         mlp_num=1, curve_config=curve_config[setting][0])

        self.cic21 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2,
                         mlp_num=1, curve_config=curve_config[setting][1])
        self.cic22 = CIC(npoint=1024, radius=0.1, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4,
                         mlp_num=1, curve_config=curve_config[setting][1])

        self.cic31 = CIC(npoint=256, radius=0.1, k=k, in_channels=128, output_channels=256, bottleneck_ratio=2,
                         mlp_num=1, curve_config=curve_config[setting][2])
        self.cic32 = CIC(npoint=256, radius=0.2, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4,
                         mlp_num=1, curve_config=curve_config[setting][2])

        self.cic41 = CIC(npoint=64, radius=0.2, k=k, in_channels=256, output_channels=512, bottleneck_ratio=2,
                         mlp_num=1, curve_config=curve_config[setting][3])
        self.cic42 = CIC(npoint=64, radius=0.4, k=k, in_channels=512, output_channels=512, bottleneck_ratio=4,
                         mlp_num=1, curve_config=curve_config[setting][3])

        self.conv0 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

    def forward(self, xyz):
        batch_size, num_points, num_dims = xyz.size()
        assert num_dims == 3, "expected BNC shape as input"
        xyz = xyz.permute(0, 2, 1)

        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)

        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)

        x = self.conv0(l4_points)
        x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x_max, x_avg), dim=1)
        return x


if __name__ == '__main__':
    data = torch.rand(32, 1024, 3).cuda()
    print("Input: ", data.shape)
    print("===> testing CurveNet ...")
    model = CurveNet().cuda()
    out = model(data)
    print("Output: ", out.shape)  # [bs, 2048]
