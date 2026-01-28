import torch
import torch.nn as nn
from fer.models.registry import available_models, make_model

# --------------------------------------------------
# Simple FLOPs counter (Conv2d + Linear only)
# --------------------------------------------------
class FlopCounter:
    def __init__(self):
        self.macs = 0
        self.handles = []

    def reset(self):
        self.macs = 0

    def conv_hook(self, m, x, y):
        n, cout, h, w = y.shape
        cin = m.in_channels
        kh, kw = m.kernel_size
        groups = m.groups
        self.macs += n * cout * h * w * (cin // groups) * kh * kw

    def linear_hook(self, m, x, y):
        batch = y.numel() // y.shape[-1]
        self.macs += batch * m.in_features * m.out_features

    def attach(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                self.handles.append(m.register_forward_hook(self.conv_hook))
            elif isinstance(m, nn.Linear):
                self.handles.append(m.register_forward_hook(self.linear_hook))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters())

def fmt(n):
    if n >= 1e9: return f"{n/1e9:.2f}G"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.2f}K"
    return str(int(n))


# --------------------------------------------------
# Main
# --------------------------------------------------
IMG = 64
IN_CH = 3
NUM_CLASSES = 7

rows = []
counter = FlopCounter()
x = torch.randn(1, IN_CH, IMG, IMG)

for name in available_models():
    try:
        model = make_model(name, num_classes=NUM_CLASSES, in_channels=IN_CH)
        params = count_params(model)

        counter.reset()
        counter.attach(model)
        with torch.no_grad():
            model(x)
        counter.detach()

        flops = 2 * counter.macs
        rows.append((name, params, flops))
    except Exception:
        rows.append((name, 0, None))

# 🔽 SORT BY PARAMETER COUNT
rows.sort(key=lambda r: r[1])

# --------------------------------------------------
# Print table
# --------------------------------------------------
print("=" * 78)
print(f"{'Rank':<5} {'Model':<32} {'Params':>12} {'FLOPs':>12}")
print("=" * 78)

for i, (name, params, flops) in enumerate(rows, 1):
    p = fmt(params)
    f = "NA" if flops is None else fmt(flops)
    print(f"{i:<5} {name:<32} {p:>12} {f:>12}")

print("=" * 78)
