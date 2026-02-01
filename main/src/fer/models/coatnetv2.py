import torch
import torch.nn as nn 
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange 

configs = {
    'coatnetv2-0': {
        'num_blocks': [2, 2, 3, 5, 2],
        'num_channels': [64, 96, 192, 384, 768],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 8,
        'block_types': ['C', 'C', 'C', 'T']
    },
    'coatnetv2-1': {
        'num_blocks': [2, 2, 6, 14, 2],
        'num_channels': [64, 96, 192, 384, 768],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 8,
        'block_types': ['C', 'C', 'C', 'T']
    },
    'coatnetv2-2': {
        'num_blocks': [2, 2, 6, 14, 2],
        'num_channels': [128, 128, 256, 512, 1024],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 8,
        'block_types': ['C', 'C', 'C', 'T']
    },
    'coatnetv2-3': {
        'num_blocks': [2, 2, 6, 14, 2],
        'num_channels': [192, 192, 384, 768, 1536],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 8,
        'block_types': ['C', 'C', 'C', 'T']
    },
    'coatnetv2-4': {
        'num_blocks': [2, 2, 12, 28, 2],
        'num_channels': [192, 192, 384, 768, 1536],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 8,
        'block_types': ['C', 'C', 'C', 'T']
    },
    'coatnetv2-5': {
        'num_blocks': [2, 2, 12, 28, 2],
        'num_channels': [192, 256, 512, 1280, 2048],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 64,
        'block_types': ['C', 'C', 'C', 'T']
    }
}

class SE(nn.Module):
    def __init__(self, channels, expansion=0.25):
        super().__init__()
        hidden = int(channels * expansion)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn='mish', ff_dropout=0.1):
        super().__init__()

        if act_fn == 'mish':
            act = nn.Mish()
        elif act_fn == 'relu':
            act = nn.ReLU(inplace=True)
        elif act_fn == 'gelu':
            act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {act_fn}")

        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            act,
            nn.Dropout(ff_dropout),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        stride = 2 if downsample else 1
        hidden_dim = int(inp * expansion)

        self.use_proj = downsample or (inp != oup)

        if self.use_proj:
            self.proj = nn.Sequential(
                nn.Conv2d(inp, oup, 1, stride=stride, bias=False),
                nn.BatchNorm2d(oup)
            )

        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            SE(hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        res = self.proj(x) if self.use_proj else x
        return res + self.conv(x)

class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid(torch.arange(self.ih),torch.arange(self.iw),indexing="ij")
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten()
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        relative_bias = self.relative_bias_table[self.relative_index.squeeze()]
        relative_bias = rearrange(
            relative_bias,
            '(n1 n2) h -> h n1 n2',
            n1=self.ih * self.iw,
            n2=self.ih * self.iw
        )
        dots = dots + relative_bias.unsqueeze(0)

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
    
class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, downsample=False):
        super().__init__()
        self.downsample = downsample

        if downsample:
            self.pool = nn.MaxPool2d(2, 2)
            self.proj = nn.Conv2d(inp, oup, 1, bias=False)

        dim = oup if downsample else inp

        self.attn = PreNorm(dim, Attention(dim, heads=heads))
        self.ff = PreNorm(dim, FeedForward(dim, dim * 4))

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool(x))

        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')

        x = x + self.attn(x)
        x = x + self.ff(x)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x

class ResidualStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


blocks = {
    'C': MBConv,
    'T': Transformer,
}

class CoAtNetV2(nn.Module):
    def __init__(self, inp_h, inp_w, in_channels, config='coatnetv2-0', num_classes=None, head_act_fn='mish', head_dropout=0.1):
        super().__init__()
        self.config = configs[config]
        block_types = self.config['block_types']
        self.s0 = self._make_stem(in_channels)
        self.s1 = self._make_block(block_types[0],
                                   inp_h, inp_w,
                                   self.config['num_channels'][0],
                                   self.config['num_channels'][1],
                                   self.config['num_blocks'][1],
                                   self.config['expand_ratio'][0],
                                   downsample_first=True)
        self.s2 = self._make_block(block_types[1], inp_h >> 1, inp_w >> 1,
                                   self.config['num_channels'][1],
                                   self.config['num_channels'][2],
                                   self.config['num_blocks'][2],
                                   self.config['expand_ratio'][1])
        self.s3 = self._make_block(block_types[2], inp_h >> 2, inp_w >> 2,
                                   self.config['num_channels'][2],
                                   self.config['num_channels'][3],
                                   self.config['num_blocks'][3],
                                   self.config['expand_ratio'][2])
        self.s4 = self._make_block(block_types[3], inp_h >> 3, inp_w >> 3,
                                   self.config['num_channels'][3],
                                   self.config['num_channels'][4],
                                   self.config['num_blocks'][4],
                                   self.config['expand_ratio'][3])
        self.include_head = num_classes is not None
        if self.include_head:
            if isinstance(num_classes, int):
                self.single_head = True
                num_classes = [num_classes]
            else:
                self.single_head = False
            self.heads = nn.ModuleList([ProjectionHead(self.config['num_channels'][-1], nc, act_fn=head_act_fn, ff_dropout=head_dropout) for nc in num_classes])
    
    def _make_stem(self, in_channels):
        return ResidualStem(in_channels, self.config["num_channels"][0])

    def _make_block(self, block_type, inp_h, inp_w, in_channels, out_channels, depth, expand_ratio, downsample_first=True):
        blocks_list = []
        cur_h, cur_w = inp_h, inp_w

        for i in range(depth):
            downsample = (i == 0 and downsample_first)
            inp_c = in_channels if i == 0 else out_channels

            if downsample:
                cur_h //= 2
                cur_w //= 2

            image_size = (cur_h, cur_w)

            if block_type == 'C':
                blocks_list.append(
                    MBConv(
                        inp=inp_c,
                        oup=out_channels,
                        image_size=image_size,
                        downsample=downsample,
                        expansion=expand_ratio
                    )
                )
            elif block_type == 'T':
                blocks_list.append(
                    Transformer(
                        inp=inp_c,
                        oup=out_channels,
                        image_size=image_size,
                        heads=self.config['n_head'],
                        downsample=downsample
                    )
                )
            else:
                raise ValueError(f"Unknown block type: {block_type}")

        return nn.Sequential(*blocks_list)
    
    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        if self.include_head:
            if self.single_head:
                return self.heads[0](x)
            return [head(x) for head in self.heads]
        return x

if __name__ == '__main__':
    import torch
    from utils import print_num_params
    from fvcore.nn import FlopCountAnalysis

    image_size = (3, 64, 64)
    config=f'coatnetv2-0'
    coatnet = CoAtNetV2(inp_h=64,inp_w=64,in_channels=3,config='coatnetv2-0',num_classes=6)
    coatnet.to('cuda:0')
    coatnet.eval()
    random_image = torch.randint(0, 256, size=(1, *image_size)).float() / 255
    with torch.no_grad():
        flops = FlopCountAnalysis(coatnet, random_image.to('cuda:0'))
    print(config)
    print(f'Approx FLOPs count: {flops.total() / 1e9:.2f}')
    print_num_params(coatnet)
