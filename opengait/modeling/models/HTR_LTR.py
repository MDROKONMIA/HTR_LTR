from typing import Sequence, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import SeparateFCs, PackSequenceWrapper, SeparateBNNecks


# -------- 3D Convolution Block --------
class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode='ltr'):
        super().__init__()
        assert mode in ['ltr', 'htr']

        if mode == 'ltr':
            k1, k2, k3 = (1,1,1),(1,3,3),(1,1,1)
            p1, p2, p3 = (0,0,0),(0,1,1),(0,0,0)
        else:
            k1, k2, k3 = (3,1,1),(1,3,3),(1,1,1)
            p1, p2, p3 = (1,0,0),(0,1,1),(0,0,0)

        mid = out_channels

        self.conv1 = nn.Conv3d(in_channels, mid, k1, padding=p1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid)

        self.conv2 = nn.Conv3d(mid, mid, k2, padding=p2, bias=False)
        self.bn2 = nn.BatchNorm3d(mid)

        self.conv3 = nn.Conv3d(mid, out_channels, k3, padding=p3, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1, bias=False)

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        if self.shortcut is not None:
            res = self.shortcut(res)

        return x + res


# -------- GeM + HPP Pooling --------
class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1e-6):
        super().__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def gem(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (1, x.size(-1))
        ).pow(1. / self.p)

    def forward(self, x):
        n, c = x.size()[:2]
        feats = []

        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            feats.append(z)

        return torch.cat(feats, -1)


# -------- HTRLTR Network --------
class HTRLTR(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_network(self, model_cfg: Dict[str, Any]):

        inner_planes: Sequence[int] = list(model_cfg['in_channels'])
        class_num = int(model_cfg['class_num'])

        self.alpha = int(model_cfg['alpha'])
        self.beta = float(model_cfg['beta'])
        self.tau = float(model_cfg['tau'])

        # Initial convolutions
        htr_conv1_out = 32
        ltr_conv1_out = max(1, int(htr_conv1_out * self.tau))

        self.s_conv1 = nn.Conv3d(
            1, ltr_conv1_out,
            kernel_size=(1,3,3),
            stride=(1,1,1),
            padding=(0,1,1),
            bias=False
        )
        self.s_bn1 = nn.BatchNorm3d(ltr_conv1_out)

        self.f_conv1 = nn.Conv3d(
            1, htr_conv1_out,
            kernel_size=(3,1,1),
            stride=(1,1,1),
            padding=(1,0,0),
            bias=False
        )
        self.f_bn1 = nn.BatchNorm3d(htr_conv1_out)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(
            kernel_size=(1,3,3),
            stride=(1,2,2),
            padding=(0,1,1)
        )

        # Channel configuration
        htr_inner = list(inner_planes)
        ltr_inner = [max(1, int(p * self.tau)) for p in inner_planes]

        # Residual blocks
        self.s_res2 = Conv3DBlock(ltr_conv1_out, ltr_inner[0], 'ltr')
        self.s_res3 = Conv3DBlock(ltr_inner[0], ltr_inner[1], 'ltr')
        self.s_res4 = Conv3DBlock(ltr_inner[1], ltr_inner[2], 'ltr')
        self.s_res5 = Conv3DBlock(ltr_inner[2], ltr_inner[3], 'ltr')
        self.s_res6 = Conv3DBlock(ltr_inner[3], ltr_inner[3], 'ltr') # presence for only Gait3D

        self.f_res2 = Conv3DBlock(htr_conv1_out, htr_inner[0], 'htr')
        self.f_res3 = Conv3DBlock(htr_inner[0], htr_inner[1], 'htr')
        self.f_res4 = Conv3DBlock(htr_inner[1], htr_inner[2], 'htr')
        self.f_res5 = Conv3DBlock(htr_inner[2], htr_inner[3], 'htr')
        self.f_res6 = Conv3DBlock(htr_inner[3], htr_inner[3], 'htr') #presence for only Gait3D

        # Lateral connections
        def lateral_op(f_c, s_c):
            return nn.Sequential(
                nn.Conv3d(
                    f_c, s_c,
                    kernel_size=(7,1,1),
                    stride=(self.alpha,1,1),
                    padding=(3,0,0),
                    bias=False
                ),
                nn.BatchNorm3d(s_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(s_c, s_c, 1, bias=False),
                nn.BatchNorm3d(s_c)
            )

        self.lateral2 = lateral_op(htr_inner[0], ltr_inner[0])
        self.lateral3 = lateral_op(htr_inner[1], ltr_inner[1])
        self.lateral4 = lateral_op(htr_inner[2], ltr_inner[2])
        self.lateral5 = lateral_op(htr_inner[3], ltr_inner[3])
        self.lateral6 = lateral_op(htr_inner[3], ltr_inner[3]) #presence for only Gait3D

        # intermediate convolution (present only for CCPG)
        self.s_conv2 = nn.Conv3d(ltr_inner[1], ltr_inner[1], kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0,0,0), bias=False)
        self.f_conv2 = nn.Conv3d(htr_inner[1], htr_inner[1], kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)

        # Pooling & heads
        self.set_pooling = PackSequenceWrapper(torch.max)

        self.HPP1 = GeMHPP(bin_num=[32])
        self.HPP2 = GeMHPP(bin_num=[32])

        head_in = ltr_inner[3] + htr_inner[3]

        self.Head0 = SeparateFCs(32, head_in, head_in)

        if 'SeparateBNNecks' in model_cfg and model_cfg['SeparateBNNecks']:
            self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
            self.Bn_head = False
        else:
            self.Bn = nn.BatchNorm1d(head_in)
            self.Head1 = SeparateFCs(32, head_in, class_num)
            self.Bn_head = True

        self._init_weights()

    # Weight initialization
    def _init_weights(self):
        for m in self.modules():

            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='leaky_relu'
                )
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # Temporal sampling for LTR stream
    def _sample_ltr(self, x):
        return x[:, :, ::self.alpha, :, :]

    def forward(self, inputs):

        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL

        sils = ipts[0]

        if sils.dim() == 4:
            sils = sils.unsqueeze(1)

        htr_in = sils
        ltr_in = self._sample_ltr(sils)

        # Initial layers
        s = self.maxpool(self.relu(self.s_bn1(self.s_conv1(ltr_in))))
        f = self.maxpool(self.relu(self.f_bn1(self.f_conv1(htr_in))))

        # Residual stages
        f = self.f_res2(f)
        s = self.s_res2(s) + self.lateral2(f)

        f = self.f_res3(f)
        s = self.s_res3(s) + self.lateral3(f)

        ## 64-->32 (present only for CCPG )
        f=self.f_conv2(self.maxpool(f))
        s=self.s_conv2(self.maxpool(s))

        f = self.f_res4(f)
        s = self.s_res4(s) + self.lateral4(f)

        f = self.f_res5(f)
        s = self.s_res5(s) + self.lateral5(f)

        ## presence only for Gait3D
        f = self.f_res6(f)
        s = self.s_res6(s) + self.lateral6(f)

        # Temporal pooling
        s_pool = self.set_pooling(s, seqL, options={"dim":2})[0]
        f_pool = self.set_pooling(f, seqL, options={"dim":2})[0]

        # HPP features
        feature1 = self.HPP1(s_pool)
        feature2 = self.HPP2(f_pool)

        feature = torch.cat([feature1, feature2], dim=1)

        gait = self.Head0(feature)

        # Classification heads
        if self.Bn_head:
            bnft = self.Bn(gait)
            logi = self.Head1(bnft)
            embed = bnft
        else:
            bnft, logi = self.BNNecks(gait)
            embed = gait

        n, _, s_frames, h, w = sils.size()

        return {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}  # absence for CCPG, Gait3D and OULP
            },
            'visual_summary': {
                'image/sils': sils.view(n*s_frames,1,h,w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
