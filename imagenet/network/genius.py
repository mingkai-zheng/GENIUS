import torch.nn as nn
from timm.models.layers import create_conv2d
from timm.models.efficientnet_blocks import ConvBnAct
from network.op import DepthwiseSeparableConv, InvertedResidual
from timm.models.efficientnet_builder import efficientnet_init_weights
from timm.models.layers.activations import HardSwish
from timm.models.layers.activations import hard_sigmoid
from functools import partial
from collections import OrderedDict
import torch.nn.functional as F


se_ratio=0.25
se_kwargs = dict(
    act_layer=nn.ReLU,
    gate_fn=hard_sigmoid,
    reduce_mid=True,
    divisor=8
)

inplace_swish = partial(HardSwish, inplace=True)

class Identity(nn.Module):
    def __init__(self, in_chs, out_chs, stride):
        super(Identity, self).__init__()
        if in_chs != out_chs or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chs),
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return x
    


class GENIUS(nn.Module):
    def __init__(self, arch=[
        [1, 0, 0, 0, 1, 0],
        [1, 0, 1, 1, 1, 0],
        [4, 0, 4, 3, 0, 0],
        [4, 3, 0, 4, 4, 0],
        [5, 0, 5, 5, 5, 0],
    ],
            dropout=0.2, drop_path_rate=0.0, norm_layer=nn.BatchNorm2d, act_layer=inplace_swish, num_classes=1000):
        super(GENIUS, self).__init__()

        self.dropout = dropout

        DSConv  = partial(DepthwiseSeparableConv, dilation=1, norm_layer=norm_layer, act_layer=act_layer, se_ratio=se_ratio, se_kwargs=se_kwargs)
        IRConv  = partial(InvertedResidual,       dilation=1, norm_layer=norm_layer, act_layer=act_layer, se_ratio=se_ratio, se_kwargs=se_kwargs)
        OPS = OrderedDict()
        OPS['id'] = lambda in_chs, out_chs, stride, dpr : Identity(in_chs, out_chs, stride)
        OPS['ir_k3_e4'] = lambda in_chs, out_chs, stride, dpr : IRConv(in_chs=in_chs, out_chs=out_chs, dw_kernel_size=3, stride=stride, exp_ratio=4, drop_path_rate=dpr)
        OPS['ir_k5_e4'] = lambda in_chs, out_chs, stride, dpr : IRConv(in_chs=in_chs, out_chs=out_chs, dw_kernel_size=5, stride=stride, exp_ratio=4, drop_path_rate=dpr)
        OPS['ir_k7_e4'] = lambda in_chs, out_chs, stride, dpr : IRConv(in_chs=in_chs, out_chs=out_chs, dw_kernel_size=7, stride=stride, exp_ratio=4, drop_path_rate=dpr)
        OPS['ir_k3_e6'] = lambda in_chs, out_chs, stride, dpr : IRConv(in_chs=in_chs, out_chs=out_chs, dw_kernel_size=3, stride=stride, exp_ratio=6, drop_path_rate=dpr)
        OPS['ir_k5_e6'] = lambda in_chs, out_chs, stride, dpr : IRConv(in_chs=in_chs, out_chs=out_chs, dw_kernel_size=5, stride=stride, exp_ratio=6, drop_path_rate=dpr)
        OPS['ir_k7_e6'] = lambda in_chs, out_chs, stride, dpr : IRConv(in_chs=in_chs, out_chs=out_chs, dw_kernel_size=7, stride=stride, exp_ratio=6, drop_path_rate=dpr)
        OPS_LIST = ['id', 'ir_k3_e4', 'ir_k5_e4', 'ir_k7_e4', 'ir_k3_e6', 'ir_k5_e6', 'ir_k7_e6']


        total_layers = 0
        for stage_i, stage_config in enumerate(arch):
            for layer_i, op_idx in enumerate(stage_config):
                if layer_i == 0 or op_idx == 0:
                    continue
                else:
                    total_layers += 1
                

        stem_size = 16
        stage_channel = [
            (16, 24),
            (24, 40),
            (40, 80),
            (80, 96),
            (96, 192),
        ]
        stage_stride = [2, 2, 2, 1, 2]

        self.conv_stem = create_conv2d(in_channels=3, out_channels=stem_size, kernel_size=3, stride=2)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer()
        self.stage0 = DSConv(in_chs=stem_size, out_chs=stem_size, dw_kernel_size=3, stride=1)
        
        current_layer = 0
        blocks = []
        for stage_i, stage_config in enumerate(arch):
            stage_block = []
            for layer_i, op_idx in enumerate(stage_config):
                op_name = OPS_LIST[op_idx]
                if layer_i > 0 and op_name == 'id':
                    continue

                if layer_i == 0:
                    in_chs, out_chs = stage_channel[stage_i]
                    stride = stage_stride[stage_i]
                    op = OPS[op_name](in_chs=in_chs, out_chs=out_chs, stride=stride, dpr=0)
                else:
                    in_chs = out_chs = stage_channel[stage_i][1]
                    stride = 1
                    op = OPS[op_name](in_chs=in_chs, out_chs=out_chs, stride=stride, dpr=current_layer / total_layers * drop_path_rate)
                    current_layer += 1
                stage_block.append(op)
            blocks.append(nn.Sequential(*stage_block))
        
        self.stage1 = blocks[0]
        self.stage2 = blocks[1]
        self.stage3 = blocks[2]
        self.stage4 = blocks[3]
        self.stage5 = blocks[4]
        self.stage6 = ConvBnAct(in_chs=192, out_chs=320, kernel_size=1, stride=1, act_layer=act_layer, norm_layer=norm_layer)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = create_conv2d(in_channels=320, out_channels=1280, kernel_size=1, bias=True)
        self.act2 = act_layer()
        self.classifier = nn.Linear(1280, out_features=num_classes)

        efficientnet_init_weights(self)

    def reset_classifier(self, num_classes):
        self.classifier = nn.Linear(1280, out_features=num_classes)
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()
        
    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.flatten(1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x
