from models.nf.nf_head import *
from models.common import *
from models.classifiers import get_feature_encoder


class HybridModel(nn.Module):
    def __init__(self, args, num_classes):
        super(HybridModel, self).__init__()
        self.in_dim = args.cla_input_dim  # feature encoder output dim
        self.enco = get_feature_encoder(args)
        self.shared_fc = nn.Sequential(
            nn.Linear(self.in_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            get_activation(args.act)
        )

        self.cls_penulti = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(512, 256, bias=False)
        )

        self.cls_head = nn.Sequential(
            nn.BatchNorm1d(256),
            get_activation(args.act),
            nn.Dropout(p=args.dropout),
            nn.Linear(256, num_classes)
        )

        self.nf_head = build_nf_head(input_dim=512)

    def forward(self, points, return_penultimate=False):
        x1 = self.enco(points)
        x2 = self.shared_fc(x1)  # [bs, 512]
        penultimate = self.cls_penulti(x2)
        if return_penultimate:
            return penultimate
        cls_logits = self.cls_head(penultimate)
        nf_logits = self.nf_head(x2)
        return cls_logits, nf_logits


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.in_dim = args.cla_input_dim  # feature encoder output dim
        self.enco = get_feature_encoder(args)
        self.shared_fc = nn.Sequential(
            nn.Linear(self.in_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            get_activation(args.act)
        )

    def forward(self, points):
        x1 = self.enco(points)  # [bs, in_dim]
        x2 = self.shared_fc(x1)  # [bs, 512]
        return x2


def build_cls_head(input_dim, num_classes, args):
    return nn.Sequential(
        nn.Linear(input_dim, 256, bias=False),
        nn.BatchNorm1d(256),
        get_activation(args.act),
        nn.Dropout(p=args.dropout),
        nn.Linear(256, num_classes)
    )
