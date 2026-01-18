from functools import partial
from collections import OrderedDict

import torch
from torch import nn

MODEL_CONFIGS = {
    "efficientnetv2-s": {
        "structure": "efficientnet_v2_s",
        "out_channels": 1280,
        "dropout": 0.2,
        "stochastic_depth": 0.2,
    },
    "efficientnetv2-m": {
        "structure": "efficientnet_v2_m",
        "out_channels": 1280,
        "dropout": 0.3,
        "stochastic_depth": 0.3,
    },
    "efficientnetv2-l": {
        "structure": "efficientnet_v2_l",
        "out_channels": 1280,
        "dropout": 0.4,
        "stochastic_depth": 0.4,
    },
    "efficientnetv2-xl": {
        "structure": "efficientnet_v2_xl",
        "out_channels": 1280,
        "dropout": 0.5,
        "stochastic_depth": 0.5,
    },
}

def get_efficientnet_v2_structure(name):
    if name == "efficientnet_v2_s":
        return [
            (1, 3, 1, 24, 24, 2, False, True),
            (4, 3, 2, 24, 48, 4, False, True),
            (4, 3, 2, 48, 64, 4, False, True),
            (4, 3, 2, 64, 128, 6, True, False),
            (6, 3, 1, 128, 160, 9, True, False),
            (6, 3, 2, 160, 256, 15, True, False),
        ]

    elif name == "efficientnet_v2_m":
        return [
            (1, 3, 1, 24, 24, 3, False, True),
            (4, 3, 2, 24, 48, 5, False, True),
            (4, 3, 2, 48, 80, 5, False, True),
            (4, 3, 2, 80, 160, 7, True, False),
            (6, 3, 1, 160, 176, 14, True, False),
            (6, 3, 2, 176, 304, 18, True, False),
            (6, 3, 1, 304, 512, 5, True, False),
        ]

    elif name == "efficientnet_v2_l":
        return [
            (1, 3, 1, 32, 32, 4, False, True),
            (4, 3, 2, 32, 64, 7, False, True),
            (4, 3, 2, 64, 96, 7, False, True),
            (4, 3, 2, 96, 192, 10, True, False),
            (6, 3, 1, 192, 224, 19, True, False),
            (6, 3, 2, 224, 384, 25, True, False),
            (6, 3, 1, 384, 640, 7, True, False),
        ]
    
    elif name == "efficientnet_v2_xl":
        return [
            (1, 3, 1, 32, 32, 4, False, True),
            (4, 3, 2, 32, 64, 8, False, True),
            (4, 3, 2, 64, 96, 8, False, True),
            (4, 3, 2, 96, 192, 16, True, False),
            (6, 3, 1, 192, 256, 24, True, False),
            (6, 3, 2, 256, 512, 32, True, False),
            (6, 3, 1, 512, 640, 8, True, False),
        ]
    
    else:
        raise ValueError(f"Structure {name} not implemented")

class ConvBNAct(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, stride, groups, norm_layer, act, conv_layer=nn.Conv2d):
        super(ConvBNAct, self).__init__(
            conv_layer(in_channel, out_channel, kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=groups, bias=False),
            norm_layer(out_channel),
            act()
        )


class SEUnit(nn.Module):
    def __init__(self, in_channel, reduction_ratio=4, act1=partial(nn.SiLU, inplace=True), act2=nn.Sigmoid):
        super(SEUnit, self).__init__()
        hidden_dim = in_channel // reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(in_channel, hidden_dim, (1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, in_channel, (1, 1), bias=True)
        self.act1 = act1()
        self.act2 = act2()

    def forward(self, x):
        return x * self.act2(self.fc2(self.act1(self.fc1(self.avg_pool(x)))))


class StochasticDepth(nn.Module):
    def __init__(self, prob, mode):
        super(StochasticDepth, self).__init__()
        self.prob = prob
        self.survival = 1.0 - prob
        self.mode = mode

    def forward(self, x):
        if self.prob == 0.0 or not self.training:
            return x
        else:
            shape = [x.size(0)] + [1] * (x.ndim - 1) if self.mode == 'row' else [1]
            return x * torch.empty(shape).bernoulli_(self.survival).div_(self.survival).to(x.device)


class MBConvConfig:
    def __init__(self, expand_ratio: float, kernel: int, stride: int, in_ch: int, out_ch: int, layers: int,
                 use_se: bool, fused: bool, act=nn.SiLU, norm_layer=nn.BatchNorm2d):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_layers = layers
        self.act = act
        self.norm_layer = norm_layer
        self.use_se = use_se
        self.fused = fused

    @staticmethod
    def adjust_channels(channel, factor, divisible=8):
        new_channel = int(channel * factor)
        divisible_channel = max(divisible, (int(new_channel + divisible / 2) // divisible) * divisible)
        divisible_channel += divisible if divisible_channel < 0.9 * new_channel else 0
        return divisible_channel


class MBConv(nn.Module):
    def __init__(self, c, sd_prob=0.0):
        super(MBConv, self).__init__()
        inter_channel = c.adjust_channels(c.in_ch, c.expand_ratio)
        block = []
        if c.fused:
            if c.expand_ratio == 1:
                block.append(('fused',ConvBNAct(c.in_ch, c.out_ch, c.kernel, c.stride, 1, c.norm_layer, c.act)))
            else:
                block.append(('fused', ConvBNAct(c.in_ch, inter_channel, c.kernel, c.stride, 1, c.norm_layer, c.act)))
                block.append(('fused_point_wise', ConvBNAct(inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity)))
        else:
            block.append(('linear_bottleneck', ConvBNAct(c.in_ch, inter_channel, 1, 1, 1, c.norm_layer, c.act)))
            block.append(('depth_wise', ConvBNAct(inter_channel, inter_channel, c.kernel, c.stride, inter_channel, c.norm_layer, c.act)))
            if c.use_se:
                block.append(('se', SEUnit(inter_channel, reduction_ratio=4)))
            block.append(('point_wise', ConvBNAct(inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity)))

        self.block = nn.Sequential(OrderedDict(block))
        self.use_skip_connection = c.stride == 1 and c.in_ch == c.out_ch
        self.stochastic_path = StochasticDepth(sd_prob, "row")

    def forward(self, x):
        out = self.block(x)
        if self.use_skip_connection:
            out = x + self.stochastic_path(out)
        return out




class EfficientNetV2(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        in_channels: int = 3,
        input_size: int = 64,
        act_layer=nn.SiLU,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Valid models: {tuple(MODEL_CONFIGS.keys())}"
            )

        cfg = MODEL_CONFIGS[model_name]

        layer_infos = [
            MBConvConfig(*layer)
            for layer in get_efficientnet_v2_structure(cfg["structure"])
        ]

        self.cur_block = 0
        self.num_blocks = sum(l.num_layers for l in layer_infos)
        self.stochastic_depth = cfg["stochastic_depth"]

        self.stem = ConvBNAct(
            in_channel=in_channels,
            out_channel=layer_infos[0].in_ch,
            kernel_size=3,
            stride=1 if input_size <= 64 else 2,
            groups=1,
            norm_layer=norm_layer,
            act=act_layer,
        )

        blocks = []
        for layer in layer_infos:
            for _ in range(layer.num_layers):
                blocks.append(
                    MBConv(layer, self._sd_prob())
                )
                layer.in_ch = layer.out_ch
                layer.stride = 1

        self.blocks = nn.Sequential(*blocks)


        self.head = nn.Sequential(
            ConvBNAct(
                layer_infos[-1].out_ch,
                cfg["out_channels"],
                kernel_size=1,
                stride=1,
                groups=1,
                norm_layer=norm_layer,
                act=act_layer,
            ),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["out_channels"], num_classes),
        )

        self._init_weights()

    def _sd_prob(self):
        prob = self.stochastic_depth * self.cur_block / self.num_blocks
        self.cur_block += 1
        return prob

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)