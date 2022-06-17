import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from models.ARPL_utils import MultiBatchNorm

def logits_entropy_loss(x):
    epsilon= 1e-20
    # softmax already applied
    predict_prob = F.softmax(x, 1)
    # predict_prob = torch.from_numpy(x)
    entropy = -predict_prob*torch.log(predict_prob + epsilon)
    return entropy.mean(1)


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU()
    elif activation.lower() == 'selu':
        return nn.SELU()
    elif activation.lower() == 'silu':
        return nn.SiLU()
    elif activation.lower() == 'hardswish':
        return nn.Hardswish()
    elif activation.lower() == 'leaky' or activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    else:
        return nn.ReLU()


def build_penultimate_proj(in_dim, p_drop, act='leakyrelu'):
    """Create a projector for models using a standard CE loss

    Args:
        in_dim (int): number of input dims
        p_drop (float): dropout probability
        act (str): activation function to apply
    """
    return nn.Sequential(
        nn.Linear(in_dim, 512, bias=False),
        nn.BatchNorm1d(512),
        get_activation(act),
        nn.Dropout(p=p_drop),
        nn.Linear(512, 256, bias=False)
    )


def build_cla_head(num_classes, p_drop, act='leakyrelu'):
    """Create a classification head for models using a standard CE loss

    Args:
        num_classes (int): number of classification outputs
        p_drop (float): dropout probability
        act (str): activation function to apply
    """

    return nn.Sequential(
        nn.BatchNorm1d(256),
        get_activation(act),
        nn.Dropout(p=p_drop),
        nn.Linear(256, num_classes)
    )


def build_hyperspherical_proj(in_dim, hidden_dim, output_dim, p_drop, act='leakyrelu'):
    """Create a projector for models using an hyperspherical features space 
    i.e.: supcon, cosface, arcface, ecc

    Args:
        in_dim (int): number of input dims
        hidden_dim (int): size of hidden layer
        output_dim (int): number of output dims 
        p_drop (float): dropout probability
        act (str): activation function to apply
    """

    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        get_activation(act),
        nn.Dropout(p=p_drop),
        nn.Linear(hidden_dim, output_dim)
    )


def cosine_sim(x1, x2, dim=1, eps=1e-8):
    """Performs cosine similarity between all vector pairs in two tensors

    Args:
        x1 (Tensor): first set of vectors
        x2 (Tensor): second set of vectors
        dim (int): dimension for normalization
        eps (float): epsilon for numerical stability

    Returns matrix of cosine similarities
    """
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)


class Penultimate_proj_ABN(nn.Module):
    """Create a projector for models using ARPL + Auxiliary Batch Norm (ABN)

    Args:
        in_dim (int): number of input dims
        p_drop (float): dropout probability
        bn_domains (int): number of batch norm domains
        act (str): activation function to apply

    """

    def __init__(self, in_dim, p_drop, bn_domains, act='leakyrelu'):
        super().__init__()

        self.l1 = nn.Linear(in_dim, 512, bias=False)
        self.bn = MultiBatchNorm(512, num_classes=bn_domains, bn_dims=1)
        self.l2 = nn.Sequential(
            get_activation(act),
            nn.Dropout(p=p_drop),
            nn.Linear(512, 256)
        )

    def forward(self, x, bn_label=0):
        x = self.l1(x)
        x = self.bn(x, bn_label)
        x = self.l2(x)
        return x


class Dist(nn.Module):
    """Computes pairwise distances between a set of samples features and a set of centers, 
    which can be stored in the module itself or passed from outside.
    """

    def __init__(self, num_classes=10, num_centers=1, feat_dim=2, init='random'):
        super(Dist, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_centers = num_centers

        if init == 'random':
            self.centers = nn.Parameter(0.1 * torch.randn(num_classes * num_centers, self.feat_dim))
        else:
            self.centers = nn.Parameter(torch.Tensor(num_classes * num_centers, self.feat_dim))
            self.centers.data.fill_(0)

    def forward(self, features, center=None, metric='l2'):
        if metric == 'l2':
            f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
            if center is None:
                c_2 = torch.sum(torch.pow(self.centers, 2), dim=1, keepdim=True)
                dist = f_2 - 2 * torch.matmul(features, torch.transpose(self.centers, 1, 0)) + torch.transpose(c_2, 1,
                                                                                                               0)
            else:
                c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
                dist = f_2 - 2 * torch.matmul(features, torch.transpose(center, 1, 0)) + torch.transpose(c_2, 1, 0)
            dist = dist / float(features.shape[1])
        else:
            if center is None:
                center = self.centers
            else:
                center = center
            dist = features.matmul(center.t())
        dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])
        dist = torch.mean(dist, dim=2)

        return dist


class ARPLoss(nn.CrossEntropyLoss):
    """ARPL loss inherited from https://github.com/iCGY96/ARPL
    """

    def __init__(self, in_features, out_features, weight_pl=0.1, temp=1.0):
        super(ARPLoss, self).__init__()
        self.weight_pl = weight_pl
        self.temp = temp
        self.Dist = Dist(num_classes=out_features, feat_dim=in_features)
        self.points = self.Dist.centers
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)

    def forward(self, x, labels=None, fake_loss=False):
        if fake_loss:
            return self.fake_loss(x)
        dist_dot_p = self.Dist(x, center=self.points, metric='dot')
        dist_l2_p = self.Dist(x, center=self.points)
        logits = dist_l2_p - dist_dot_p

        if labels is None: return logits
        loss = F.cross_entropy(logits / self.temp, labels)

        # batch of reciprocal points. For each sample in current batch it contains it's reciprocal point
        center_batch = self.points[labels, :]
        # compute eucl distance between each sample and it's reciprocal point, (mean on dimensions)
        _dis_known = (x - center_batch).pow(2).mean(1)
        # target = 1 means that first input (radius), should be larger than the second (distance)
        target = torch.ones(_dis_known.size()).cuda()
        # this is a ranking loss. It pushes for the obtainment of the desired ranking
        # equivalend to max(_dis_known - (radius -1), 0)
        # the final margin is (radius - 1)
        loss_r = self.margin_loss(self.radius, _dis_known, target)

        loss = loss + self.weight_pl * loss_r

        return logits, loss

    def fake_loss(self, x):
        # used for entropy maximization on fake data)
        logits = self.Dist(x, center=self.points)
        prob = F.softmax(logits, dim=1)
        loss = (prob * torch.log(prob)).sum(1).mean().exp()

        return loss


class MarginCosineProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        cosine = cosine_sim(input, self.weight)
        if label is None:
            return cosine
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = self.s * (cosine - one_hot * self.m)
        return output, cosine

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class SubcenterArcMarginProduct(nn.Module):
    r"""Modified implementation from https://github.com/ronghuaiyang/arcface-pytorch/blob/47ace80b128042cd8d2efd408f55c5a3e156b032/models/metrics.py#L10
        """

    def __init__(self, in_features, out_features, K=1, s=30.0, m=0.50, easy_margin=False):
        """
        Implementation of Arc-Face and sub center Arc Face from https://github.com/vladimirstarygin/Subcenter-ArcFace-Pytorch
        Easy margin explanation: https://github.com/ronghuaiyang/arcface-pytorch/issues/24
        
        Args:
            K (int) number of centers for each class. With K = 1 we have a standard ArcFace. 
        """
        super(SubcenterArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.K = K
        self.weight = nn.Parameter(torch.FloatTensor(out_features * self.K, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = cosine_sim(input, self.weight)

        if self.K > 1:
            cosine = torch.reshape(cosine, (-1, self.out_features, self.K))
            cosine, _ = torch.max(cosine, axis=2)

        if label is None:
            return cosine

        sine = torch.sqrt(
            (1.0 - torch.pow(cosine, 2)).clamp(1e-06, 1)  # may 20th - added epsilon
        )
        # cos(theta+m) = cos(theta)cos(m) - sin(theta)sin(m) 
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output, cosine


def convert_model_state(old_state_dict, new_state_dict):
    """Convert a state dict from old classifier structure to the new one. 
    This is a transitional function. It should become useless when we substitute all models 

    Args:
        old_state_dict: model state dict following old structure
        new_state_dict: model state dict following new structure 
    """

    # first of all we check if there is something to do
    good = True
    for k in old_state_dict.keys():
        if k not in new_state_dict:
            good = False
            break
    if good:
        return old_state_dict
    print("Model state changes detected. Converting dict")

    tmp_state_dict = {}

    # if the old state dict contains a key "fc" it means it was trained with a face loss 
    fc_found = False
    for k in old_state_dict.keys():
        if k.startswith("fc"):
            fc_found = True
            break

    if fc_found:
        # model from old cosface_loss branch 
        for k in old_state_dict.keys():
            if k.startswith("enco"):
                new_k = k
            elif k.startswith("head.0") or k.startswith("head.1") or k.startswith("head.4"):
                new_k = k.replace("head", "penultimate")
            elif k.startswith("fc"):
                new_k = k.replace("fc", "head")
            else:
                raise NotImplementedError(f"Unknown key: {k}")

            tmp_state_dict[new_k] = old_state_dict[k]

    else:
        # model from old main branch 
        for k in old_state_dict.keys():
            if k.startswith("enco"):
                new_k = k
            elif k.startswith("head.0") or k.startswith("head.1") or k.startswith("head.4"):
                new_k = k.replace("head", "penultimate")
            elif k.startswith("head.5") or k.startswith("head.8"):
                new_k = k.replace("5", "0").replace("8", "3")
            else:
                raise NotImplementedError(f"Unknown key: {k}")
            tmp_state_dict[new_k] = old_state_dict[k]
    return tmp_state_dict
