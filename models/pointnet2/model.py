import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from typing import Optional, Tuple
from models.ARPL_utils import MultiBatchNorm
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
from pointnet2_ops import pointnet2_utils


class Pointnet2SSG(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Classification network

        Parameters
        ----------
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=3, use_xyz=True):
        super(Pointnet2SSG, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[input_channels, 64, 64, 128],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(mlp=[256, 256, 512, 1024], use_xyz=use_xyz)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """

        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        features = features.squeeze(-1)
        return features


class Pointnet2MSG(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Classification network

        Parameters
        ----------
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=3, use_xyz=True):
        super(Pointnet2MSG, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[
                    [input_channels, 32, 32, 64],
                    [input_channels, 64, 64, 128],
                    [input_channels, 64, 96, 128],
                ],
                use_xyz=use_xyz,
            )
        )

        input_channels = 64 + 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(mlp=[128 + 256 + 256, 256, 512, 1024], use_xyz=use_xyz)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        features = features.squeeze(-1)
        return features


def get_pn2_ssg_encoder(input_channels=3, use_xyz=True):
    return Pointnet2SSG(input_channels=input_channels, use_xyz=use_xyz)


def get_pn2_msg_encoder(input_channels=3, use_xyz=True):
    return Pointnet2MSG(input_channels=input_channels, use_xyz=use_xyz)

def convert_recursive(module, bn_labels=2):

    for child_name, child in module.named_children():
        if isinstance(child, nn.modules.batchnorm.BatchNorm2d):
            print(f"Converting BN")
            old_mod = getattr(module, child_name)
            new_mod = MultiBatchNorm(old_mod.num_features, num_classes=bn_labels, bn_dims=2)
            setattr(module, child_name, new_mod)
        else:
            convert_recursive(child, bn_labels=bn_labels)

def convert_pn2_abn(base_enco, bn_labels=2):
    # first of all we convert BN modules with MultiBN
    convert_recursive(base_enco, bn_labels=2)

    # substitute base_enco forward
    def base_enco_forward(self, pointcloud, bn_label=0):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
            bn_label (int): identifier of the bn to use
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features, bn_label=bn_label)

        features = features.squeeze(-1)
        return features

    base_enco.forward = types.MethodType(base_enco_forward, base_enco)

    # substitute PointnetSAModuleBase forward 
    def SA_Module_forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor], bn_label=0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        bn_label (int): identifier of the bn to use

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
        )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            # new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            for l in self.mlps[i]:
                if isinstance(l, MultiBatchNorm):
                    new_features = l(new_features, bn_label)
                else:
                    new_features = l(new_features)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)

    for module in base_enco.SA_modules:
        module.forward = types.MethodType(SA_Module_forward, module)

    return base_enco

if __name__ == '__main__':
    data = torch.rand(8, 1024, 3).cuda()
    print("Input: ", data.shape)
    print("===> testing PN2 SSG ...")
    model = get_pn2_ssg_encoder(input_channels=0, use_xyz=True).cuda()
    out = model(data)
    print("Output: ", out.shape)  # [bs, 2048]
    print("\n\n")
    data = torch.rand(8, 1024, 3).cuda()
    print("Input: ", data.shape)
    print("===> testing PN2 MSG ...")
    model = get_pn2_msg_encoder(input_channels=0, use_xyz=True).cuda()
    out = model(data)
    print("Output: ", out.shape)  # [bs, 2048]
