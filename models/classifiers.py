from models.common import *
from models import *


def get_feature_encoder(args):
    if args.ENCO_NAME.lower() == 'dgcnn':
        return DGCNN(k=args.k, emb_dims=args.emb_dims)
    elif args.ENCO_NAME.lower() == 'dgcnnabn':
        return DGCNNABN(k=args.k, emb_dims=args.emb_dims)
    elif args.ENCO_NAME.lower() == 'curvenet':
        return CurveNet(k=args.k)
    elif args.ENCO_NAME.lower() == 'gdanet':
        return GDANET()
    elif args.ENCO_NAME.lower() == 'rscnn_ssn':
        return RSCNN_SSN()
    elif args.ENCO_NAME.lower() == 'pct':
        return PCT(emb_dims=args.emb_dims)
    elif args.ENCO_NAME.lower() == 'pointmlp':
        return pointMLP()
    elif args.ENCO_NAME.lower() == 'pointmlpelite':
        return pointMLPElite()
    elif args.ENCO_NAME.lower() == 'pn2-ssg':
        return get_pn2_ssg_encoder(input_channels=0, use_xyz=True)
    elif args.ENCO_NAME.lower() == 'pn2-msg':
        return get_pn2_msg_encoder(input_channels=0, use_xyz=True)
    elif args.ENCO_NAME.lower() == 'pn2-msgabn':
        base_enco = get_pn2_msg_encoder(input_channels=0, use_xyz=True)
        return convert_pn2_abn(base_enco)
    else:
        raise ValueError("Unknown encoder")


class Classifier(nn.Module):
    def __init__(self, args, num_classes, loss="CE", cs=False):
        super(Classifier, self).__init__()
        self.in_dim = args.cla_input_dim  # classification head input dim
        self.cs = cs  # for ARPL + CS
        if self.cs:
            args.ENCO_NAME += "ABN"  # load the ABN version of feature encoder

        # encoder: [B,N,3] -> [B,C_In]
        self.enco = get_feature_encoder(args)
        print(f"Clf - feature encoder: {args.ENCO_NAME}")
        print(f"Clf Head - "
              f"num classes: {num_classes}, input dim: {self.in_dim}, act: {args.act}, dropout: {args.dropout}")

        if loss == "CE" or loss == "CE_ls":
            self.penultimate = build_penultimate_proj(self.in_dim, args.dropout, act=args.act)
            self.head = build_cla_head(num_classes, args.dropout, act=args.act)
        elif loss == "cosface":
            self.penultimate = build_hyperspherical_proj(self.in_dim, 512, 256, p_drop=args.dropout, act=args.act)
            self.head = MarginCosineProduct(256, num_classes)
        elif loss == "cosine":
            self.penultimate = build_hyperspherical_proj(self.in_dim, 512, 256, p_drop=args.dropout, act=args.act)
            self.head = MarginCosineProduct(256, num_classes, m=0)
        elif loss == "arcface":
            self.penultimate = build_hyperspherical_proj(self.in_dim, 512, 256, p_drop=args.dropout, act=args.act)
            self.head = SubcenterArcMarginProduct(256, num_classes, K=1, easy_margin=False)
        elif loss == "subcenter_arcface":
            self.penultimate = build_hyperspherical_proj(self.in_dim, 512, 256, p_drop=args.dropout, act=args.act)
            self.head = SubcenterArcMarginProduct(256, num_classes, K=3, easy_margin=False)
        elif loss == "ARPL":
            if self.cs:
                self.penultimate = Penultimate_proj_ABN(self.in_dim, p_drop=args.dropout, bn_domains=2)
            else:
                self.penultimate = build_hyperspherical_proj(self.in_dim, 512, 256, p_drop=args.dropout, act=args.act)
            self.head = ARPLoss(256, num_classes)
        else:
            raise NotImplementedError(f"Unknown loss type: {loss}")

    def forward(self, x, labels=None, return_penultimate=False, bn_label=0, fake_loss=False):
        if self.cs:
            feat = self.enco(x, bn_label)
            penultimate = self.penultimate(feat, bn_label)
        else:
            feat = self.enco(x)
            penultimate = self.penultimate(feat)

        if return_penultimate:
            return penultimate

        if fake_loss:
            return self.head(penultimate, fake_loss=fake_loss)

        if labels is not None:
            logits = self.head(penultimate, labels)
        else:
            logits = self.head(penultimate)

        return logits
