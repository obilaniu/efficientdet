from itertools import chain
from functools import partial
from pathlib   import Path

import math, pdb, os, sys, time
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import config as cfg
from log.logger import logger
from utils.utils import download_model_weights
from utils.tools import variance_scaling_


from model.efficientnet.utils import (
    round_filters,
    round_repeats,
    drop_connect,
    Conv2dStaticSamePadding,
    BlockArgs,
    GlobalParams,
)
from model.efficientnet.efficientnet import MBConvBlock


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvModule(nn.Module):
    """ Regular Convolution with BatchNorm """
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DepthWiseSeparableConvModuleB3(nn.Module):
    """ DepthWise Separable Convolution with BatchNorm and ReLU activation """
    def __init__(self, in_channels, out_channels, bath_norm=True, relu=True, bias=False):
        super().__init__()
        self.conv_dw = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                 padding=1, groups=in_channels, bias=False)
        self.conv_pw = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                 padding=0, bias=bias)

        self.bn = None if not bath_norm else \
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.act = None if not relu else Swish()

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class MaxPool2dSamePadB3(nn.MaxPool2d):
    """ TensorFlow-like 2D Max Pooling with same padding """

    PAD_VALUE: float = -float('inf')

    def __init__(self, kernel_size: int, stride=1, padding=0,
                 dilation=1, ceil_mode=False, count_include_pad=True, extra_pad=(0,0)):
        assert padding == 0, 'Padding in MaxPool2d Same Padding should be zero'

        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        padding = (padding, padding)
        dilation = (dilation, dilation)
        self.extra_pad = extra_pad

        super().__init__(kernel_size, stride, padding,
                         dilation, ceil_mode, count_include_pad)

    def forward(self, x):
        pad_h, pad_w = self.extra_pad
        #h, w = x.size()[-2:]
        #pad_h = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] + \
        #        (self.kernel_size[0] - 1) * self.dilation[0] + 1 - h
        #pad_w = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] + \
        #        (self.kernel_size[1] - 1) * self.dilation[1] + 1 - w

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                          pad_h - pad_h // 2], value=self.PAD_VALUE)

        x = F.max_pool2d(x, self.kernel_size, self.stride,
                         self.padding, self.dilation, self.ceil_mode)
        return x


class HeadNetB3(nn.Module):
    """ Box Regression and Classification Nets """
    def __init__(self, n_features, out_channels, n_repeats, num_levels=5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(n_repeats):
            self.convs.append(DepthWiseSeparableConvModuleB3(n_features, n_features,
                                                             bath_norm=False, relu=False))
            bn_levels = nn.ModuleList()
            for _ in range(num_levels):
                bn = nn.BatchNorm2d(n_features, eps=1e-3, momentum=0.01)
                bn_levels.append(bn)
            self.bns.append(bn_levels)

        self.act = Swish()
        self.head = DepthWiseSeparableConvModuleB3(n_features, out_channels,
                                                   bath_norm=False, relu=False, bias=True)

    def forward(self, inputs):
        outs = []

        for f_idx, f_map in enumerate(inputs):
            for conv, bn in zip(self.convs, self.bns):
                f_map = conv(f_map)
                f_map = bn[f_idx](f_map)
                f_map = self.act(f_map)
            outs.append(self.head(f_map))

        return tuple(outs)


class BiFPNB3(nn.Module):
    """
    BiFPN block.
    Depending on its order, it either accepts
    seven feature maps (if this block is the first block in FPN) or
    otherwise five feature maps from the output of the previous BiFPN block
    """

    EPS: float = 1e-04
    REDUCTION_RATIO: int = 2

    def __init__(self, n_channels):
        super().__init__()

        self.conv_4_td = DepthWiseSeparableConvModuleB3(n_channels, n_channels, relu=False)
        self.conv_5_td = DepthWiseSeparableConvModuleB3(n_channels, n_channels, relu=False)
        self.conv_6_td = DepthWiseSeparableConvModuleB3(n_channels, n_channels, relu=False)

        self.weights_4_td = nn.Parameter(torch.ones(2))
        self.weights_5_td = nn.Parameter(torch.ones(2))
        self.weights_6_td = nn.Parameter(torch.ones(2))

        self.conv_3_out = DepthWiseSeparableConvModuleB3(n_channels, n_channels, relu=False)
        self.conv_4_out = DepthWiseSeparableConvModuleB3(n_channels, n_channels, relu=False)
        self.conv_5_out = DepthWiseSeparableConvModuleB3(n_channels, n_channels, relu=False)
        self.conv_6_out = DepthWiseSeparableConvModuleB3(n_channels, n_channels, relu=False)
        self.conv_7_out = DepthWiseSeparableConvModuleB3(n_channels, n_channels, relu=False)

        self.weights_3_out = nn.Parameter(torch.ones(2))
        self.weights_4_out = nn.Parameter(torch.ones(3))
        self.weights_5_out = nn.Parameter(torch.ones(3))
        self.weights_6_out = nn.Parameter(torch.ones(3))
        self.weights_7_out = nn.Parameter(torch.ones(2))

        self.upsample = lambda x: F.interpolate(x, scale_factor=self.REDUCTION_RATIO)
        self.downsample = MaxPool2dSamePadB3(self.REDUCTION_RATIO + 1, self.REDUCTION_RATIO, extra_pad=(1,1))

        self.act = Swish()

    def forward(self, features):
        if len(features) == 5:
            p_3, p_4, p_5, p_6, p_7 = features
            p_4_2, p_5_2 = None, None
        else:
            p_3, p_4, p_4_2, p_5, p_5_2, p_6, p_7 = features

        # Top Down Path
        p_6_td = self.conv_6_td(
            self._fuse_features(
                weights=self.weights_6_td,
                features=[p_6, self.upsample(p_7)]
            )
        )
        p_5_td = self.conv_5_td(
            self._fuse_features(
                weights=self.weights_5_td,
                features=[p_5, self.upsample(p_6_td)]
            )
        )
        p_4_td = self.conv_4_td(
            self._fuse_features(
                weights=self.weights_4_td,
                features=[p_4, self.upsample(p_5_td)]
            )
        )

        p_4_in = p_4 if p_4_2 is None else p_4_2
        p_5_in = p_5 if p_5_2 is None else p_5_2

        # Out
        p_3_out = self.conv_3_out(
            self._fuse_features(
                weights=self.weights_3_out,
                features=[p_3, self.upsample(p_4_td)]
            )
        )
        p_4_out = self.conv_4_out(
            self._fuse_features(
                weights=self.weights_4_out,
                features=[p_4_in, p_4_td, self.downsample(p_3_out)]
            )
        )
        p_5_out = self.conv_5_out(
            self._fuse_features(
                weights=self.weights_5_out,
                features=[p_5_in, p_5_td, self.downsample(p_4_out)]
            )
        )
        p_6_out = self.conv_6_out(
            self._fuse_features(
                weights=self.weights_6_out,
                features=[p_6, p_6_td, self.downsample(p_5_out)]
            )
        )
        p_7_out = self.conv_7_out(
            self._fuse_features(
                weights=self.weights_7_out,
                features=[p_7, self.downsample(p_6_out)]
            )
        )

        return (p_3_out, p_4_out, p_5_out, p_6_out, p_7_out)

    def _fuse_features(self, weights, features):
        weights = F.relu(weights)
        num = sum([w * f for w, f in zip(weights, features)])
        det = sum(weights) + self.EPS
        x = self.act(num / det)
        return x


class ChannelAdjusterB3(nn.Module):
    """ Adjusts number of channels before BiFPN via 1x1 conv layers
    Creates P3, P4, P4_2, P5, P5_2, P6 and P7 feature maps  for use in BiFPN """
    def __init__(self, in_channels: list, out_channels: int):
        super().__init__()
        assert isinstance(in_channels, list), 'in_channels should be a list'
        assert isinstance(out_channels, int), 'out_channels should be an integer'

        self.convs = nn.ModuleList()
        self.convs.append(ConvModule(in_channels[0], out_channels))
        self.convs.append(ConvModule(in_channels[1], out_channels))
        self.convs.append(ConvModule(in_channels[1], out_channels))
        self.convs.append(ConvModule(in_channels[2], out_channels))
        self.convs.append(ConvModule(in_channels[2], out_channels))
        self.p5_to_p6 = nn.Sequential(
            ConvModule(in_channels[2], out_channels),
            MaxPool2dSamePadB3(3, 2, extra_pad=(1,1))
        )
        self.p6_to_p7 = MaxPool2dSamePadB3(3, 2, extra_pad=(1,1))

    def forward(self, features):
        """ param: features: a list of P3, P4, P5 feature maps from backbone
            returns: outs: P3, P4, P4_2, P5, P5_2, P6, P7 feature maps """
        p3, p4, p5 = features
        p6 = self.p5_to_p6(p5)
        p7 = self.p6_to_p7(p6)
        return (self.convs[0](p3),
                self.convs[1](p4),
                self.convs[2](p4),
                self.convs[3](p5),
                self.convs[4](p5), p6, p7)

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se  = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = partial(Conv2dStaticSamePadding, image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0         = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,
            # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1,int(self._block_args.input_filters*self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2          = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish        = Swish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(
                self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = Swish()


class EffNetB3(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    """

    def __init__(self, override_params=None):
        super().__init__()
        blocks_args = [
            BlockArgs(kernel_size=3, num_repeat=1, input_filters=32,  output_filters=16,  expand_ratio=1, id_skip=True, stride=[1], se_ratio=0.25),
            BlockArgs(kernel_size=3, num_repeat=2, input_filters=16,  output_filters=24,  expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25),
            BlockArgs(kernel_size=5, num_repeat=2, input_filters=24,  output_filters=40,  expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25),
            BlockArgs(kernel_size=3, num_repeat=3, input_filters=40,  output_filters=80,  expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25),
            BlockArgs(kernel_size=5, num_repeat=3, input_filters=80,  output_filters=112, expand_ratio=6, id_skip=True, stride=[1], se_ratio=0.25),
            BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25),
            BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320, expand_ratio=6, id_skip=True, stride=[1], se_ratio=0.25),
        ]

        # note: all models have drop connect rate = 0.2
        w, d, s, p = 1.2, 1.4, 300, 0.3
        global_params = GlobalParams(
            batch_norm_momentum=0.99,
            batch_norm_epsilon=1e-3,
            dropout_rate=p,
            drop_connect_rate=0.2,
            num_classes=1000,
            width_coefficient=w,
            depth_coefficient=d,
            depth_divisor=8,
            min_depth=None,
            image_size=s,
        )
        if override_params:
            # ValueError will be raised here if override_params has fields not included in global_params.
            global_params = global_params._replace(**override_params)

        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = partial(Conv2dStaticSamePadding, image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters,
                                            self._global_params),
                output_filters=round_filters(block_args.output_filters,
                                             self._global_params),
                num_repeat=round_repeats(block_args.num_repeat,
                                         self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(
                    MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = Swish()

    def forward(self, inputs):
        raise NotImplementedError("Not meant to be used directly!")

    @classmethod
    def from_name(cls, override_params=None):
        return cls(override_params)

    @classmethod
    def from_pretrained(cls, advprop=False, num_classes=1000, in_channels=3):
        model = cls.from_name(override_params={'num_classes': num_classes})
        if in_channels != 3:
            Conv2d = partial(Conv2dStaticSamePadding, image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model

    @classmethod
    def get_image_size(cls):
        return 300


class EfficientNetB3(nn.Module):
    """ Backbone Wrapper """
    def __init__(self):
        super().__init__()
        model = EffNetB3.from_pretrained()
        del model._bn1, model._conv_head, model._fc
        del model._avg_pooling, model._dropout
        self.model = model

    def forward(self, x):
        dcr = float(self.model._global_params.drop_connect_rate or 0)
        Lb = len(self.model._blocks)
        x = self.model._swish(self.model._bn0(self.model._conv_stem(x)))
        x = self.model._blocks[ 0](x, drop_connect_rate=dcr* 0/Lb)
        x = self.model._blocks[ 1](x, drop_connect_rate=dcr* 1/Lb)
        x = self.model._blocks[ 2](x, drop_connect_rate=dcr* 2/Lb)
        x = self.model._blocks[ 3](x, drop_connect_rate=dcr* 3/Lb)
        x = self.model._blocks[ 4](x, drop_connect_rate=dcr* 4/Lb)
        x = self.model._blocks[ 5](x, drop_connect_rate=dcr* 5/Lb)
        x = self.model._blocks[ 6](x, drop_connect_rate=dcr* 6/Lb)
        x = self.model._blocks[ 7](x, drop_connect_rate=dcr* 7/Lb)
        x7 = x
        x = self.model._blocks[ 8](x, drop_connect_rate=dcr* 8/Lb)
        x = self.model._blocks[ 9](x, drop_connect_rate=dcr* 9/Lb)
        x = self.model._blocks[10](x, drop_connect_rate=dcr*10/Lb)
        x = self.model._blocks[11](x, drop_connect_rate=dcr*11/Lb)
        x = self.model._blocks[12](x, drop_connect_rate=dcr*12/Lb)
        x = self.model._blocks[13](x, drop_connect_rate=dcr*13/Lb)
        x = self.model._blocks[14](x, drop_connect_rate=dcr*14/Lb)
        x = self.model._blocks[15](x, drop_connect_rate=dcr*15/Lb)
        x = self.model._blocks[16](x, drop_connect_rate=dcr*16/Lb)
        x = self.model._blocks[17](x, drop_connect_rate=dcr*17/Lb)
        x17 = x
        x = self.model._blocks[18](x, drop_connect_rate=dcr*18/Lb)
        x = self.model._blocks[19](x, drop_connect_rate=dcr*19/Lb)
        x = self.model._blocks[20](x, drop_connect_rate=dcr*20/Lb)
        x = self.model._blocks[21](x, drop_connect_rate=dcr*21/Lb)
        x = self.model._blocks[22](x, drop_connect_rate=dcr*22/Lb)
        x = self.model._blocks[23](x, drop_connect_rate=dcr*23/Lb)
        x = self.model._blocks[24](x, drop_connect_rate=dcr*24/Lb)
        x = self.model._blocks[25](x, drop_connect_rate=dcr*25/Lb)
        x25 = x
        return (x7,x17,x25)

    def get_channels_list(self):
        return [48, 136, 384]


class EfficientDetD3(nn.Module):
    def __init__(self):
        super().__init__()
        self.NAME       = 'efficientdet-d3'
        self.BACKBONE   = 'efficientnet-b3'
        self.IMAGE_SIZE = 896
        self.W_BIFPN    = 160
        self.D_BIFPN    = 6
        self.D_CLASS    = 4
        self.NUM_ANCHORS= 3*3
        self.NUM_CLASSES= 90
        self.backbone   = EfficientNetB3()
        self.adjuster   = ChannelAdjusterB3(self.backbone.get_channels_list(), self.W_BIFPN)
        self.bifpn      = nn.Sequential(*[BiFPNB3(self.W_BIFPN) for _ in range(self.D_BIFPN)])
        self.regresser  = HeadNetB3(n_features=self.W_BIFPN, out_channels=self.NUM_ANCHORS*4,                n_repeats=self.D_CLASS)
        self.classifier = HeadNetB3(n_features=self.W_BIFPN, out_channels=self.NUM_ANCHORS*self.NUM_CLASSES, n_repeats=self.D_CLASS)

    def forward(self, x):
        x = self.backbone(x)
        x = self.adjuster(x)
        x = self.bifpn(x)
        cls_outputs = self.classifier(x)
        box_outputs = self.regresser(x)
        return cls_outputs + box_outputs

    @staticmethod
    def from_name(WEIGHTS_PATH=Path('./weights')):
        """ Interface for model prepared to train on COCO """
        name = "efficientdet-d3"
        cfg.MODEL.choose_model(name)  # DELETE ME WHEN POSSIBLE!
        model = EfficientDetD3()
        BACKBONE_WEIGHTS = WEIGHTS_PATH / f'{model.BACKBONE}.pth'

        if not BACKBONE_WEIGHTS.exists():
            logger(f'Downloading backbone {model.BACKBONE}...')
            download_model_weights(model.BACKBONE, BACKBONE_WEIGHTS)

        model._load_backbone(BACKBONE_WEIGHTS)
        model._initialize_weights()
        return model

    @staticmethod
    def from_pretrained(WEIGHTS_PATH=Path('./weights')):
        """ Interface for pre-trained model """
        name = "efficientdet-d3"
        cfg.MODEL.choose_model(name)  # DELETE ME WHEN POSSIBLE!
        model = EfficientDetD3()
        WEIGHTS = WEIGHTS_PATH / f'{model.NAME}.pth'

        if not WEIGHTS.exists():
            logger(f'Downloading pre-trained {model.NAME}...')
            download_model_weights(model.NAME, WEIGHTS)

        model._load_weights(WEIGHTS)
        return model

    def _initialize_weights(self):
        """ Initialize Model Weights before training from scratch """
        for module in chain(self.adjuster.modules(),
                            self.bifpn.modules()):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        for module in chain(self.regresser.modules(),
                            self.classifier.modules()):
            if isinstance(module, nn.Conv2d):
                variance_scaling_(module.weight)
            if isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.zeros_(self.regresser.head.conv_pw.bias)
        nn.init.constant_(self.classifier.head.conv_pw.bias, -np.log((1 - 0.01) / 0.01))

    def _load_backbone(self, path):
        self.backbone.model.load_state_dict(torch.load(path), strict=False)
        logger(f'Loaded backbone checkpoint {path}')

    def _load_weights(self, path):
        self.load_state_dict(torch.load(path))
        logger(f'Loaded checkpoint {path}')
