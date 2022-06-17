import torch
import torch.nn as nn
from models.RSCNN.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG


# Relation-Shape CNN: Single-Scale Neighborhood
class RSCNN_SSN(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=0, relation_prior=1, use_xyz=True):
        super().__init__()

        print(f"RSCNN_SSN input_channels: {input_channels}")
        print(f"CurveNet relation_prior: {relation_prior}")
        print(f"CurveNet use_xyz: {use_xyz}")

        self.SA_modules = nn.ModuleList()

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.23],
                nsamples=[48],
                mlps=[[input_channels, 128]],
                first_layer=True,
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.32],
                nsamples=[64],
                mlps=[[128, 512]],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )

        self.SA_modules.append(
            # global convolutional pooling
            PointnetSAModule(
                nsample=128,
                mlp=[512, 1024],
                use_xyz=use_xyz
            )
        )

        # self.FC_layer = nn.Sequential(
        #     pt_utils.FC(1024, 512, activation=nn.ReLU(inplace=True), bn=True),
        #     nn.Dropout(p=0.5),
        #     pt_utils.FC(512, 256, activation=nn.ReLU(inplace=True), bn=True),
        #     nn.Dropout(p=0.5),
        #     pt_utils.FC(256, num_classes, activation=None)
        # )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
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
        batch_size, num_points, num_dims = pointcloud.size()
        assert num_dims == 3, "expected BNC shape as input"
        xyz, features = self._break_up_pc(pointcloud)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        features = features.view(batch_size, -1)
        return features


if __name__ == '__main__':
    data = torch.rand(32, 1024, 3).cuda()
    print("Input: ", data.shape)
    print("===> testing RSCNN_SSN ...")
    model = RSCNN_SSN().cuda()
    out = model(data)
    print("Output: ", out.shape)  # [bs, 1024]
