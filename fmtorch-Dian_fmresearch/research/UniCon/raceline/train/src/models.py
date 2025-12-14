import math, torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple
from fmtorch.models import UNetModel  # 你已安装 fmtorch
import numpy as np

# ---------- small registry ----------
REGISTRY = {}
def register(name): 
    def deco(cls): REGISTRY[name]=cls; return cls
    return deco

# ---------- utils blocks ----------
class Mish(nn.Module):
    def forward(self, x): return x * torch.tanh(F.softplus(x))

class ResidualFC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.act = Mish()
    def forward(self, x):
        y = self.act(self.fc1(x)); y = self.fc2(y)
        return self.act(y + x)

# ---------- CBAM (通道+空间) ----------
class SE(nn.Module):
    def __init__(self, C, reduction=16):
        super().__init__()
        hidden = max(1, C // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(C, hidden, 1, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(hidden, C, 1, bias=False),
                                 nn.Sigmoid())
    def forward(self, x):
        w = self.mlp(self.avg(x))
        return x * w

class CBAM(nn.Module):
    def __init__(self, C, reduction=16, k=7):
        super().__init__()
        self.se = SE(C, reduction)
        pad = (k - 1) // 2
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, k, padding=pad, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.se(x)
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        mask = self.spatial(torch.cat([avg, mx], dim=1))
        return x * mask

# ---------- time embedding ----------
class SinCosTime(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Sequential(
            nn.Linear(out_dim, 512), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, 64)    # 与 8x8 flatten 对齐
        )
    def forward(self, t: torch.Tensor):
        half = self.out_dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device) * (math.log(10000.0)/(half-1)))
        ang = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        if self.out_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(emb.size(0), 1, device=emb.device)], dim=1)
        return self.proj(emb)  # (B,64)

# ---------- fusion: FiLM ----------
class FiLM2d(nn.Module):
    def __init__(self, in_feat=64, C=16):
        super().__init__()
        self.gamma = nn.Linear(in_feat, C)
        self.beta  = nn.Linear(in_feat, C)
    def forward(self, x_img, fuse_feat):  # x_img:(B,C,H,W), fuse_feat:(B,in_feat)
        B, C, H, W = x_img.shape
        g = self.gamma(fuse_feat).view(B, C, 1, 1)
        b = self.beta(fuse_feat ).view(B, C, 1, 1)
        return x_img * (1 + g) + b

# ---------- heads ----------
class Upsampler(nn.Module):
    def __init__(self, in_dim, out_shape=(16,8,8), width=512, depth=4):
        super().__init__()
        out_dim = int(np.prod(out_shape))
        layers = [nn.Linear(in_dim, width), Mish()]
        for _ in range(depth-2): layers.append(ResidualFC(width))
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)
        self.out_shape = out_shape
    def forward(self, x):
        y = self.net(x); y = self.norm(y)
        return y.view(x.size(0), *self.out_shape)

class Downsampler(nn.Module):
    def __init__(self, in_shape=(16,8,8), out_dim=10):
        super().__init__()
        in_dim = int(np.prod(in_shape))
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), Mish(),
            ResidualFC(512),
            nn.Linear(512, out_dim)
        )
        self.in_shape = in_shape
    def forward(self, x):  # x:(B,C,H,W)
        return self.net(x.flatten(1))

# ---------- 主模型（带可选 CBAM） ----------
@register("flow_unet")
class FlowUNet(nn.Module):
    """
    y_dim:  目标向量维度（flatten(actions, states)）
    cond_dim: 条件维度（x0s）
    """
    def __init__(self, y_dim: int, cond_dim: int,
                 image_channels=16, image_hw=8,
                 up_width=512, up_depth=4,
                 time_dim=256, unet_channels=32, num_res_blocks=1,
                 attn_resolutions=(4,), enable_cbam=True, cbam_reduction=16, cbam_kernel=3,
                 fusion="film", dropout=0.1):
        super().__init__()
        self.y_dim, self.cond_dim = y_dim, cond_dim
        C, H, W = image_channels, image_hw, image_hw

        self.ups = Upsampler(y_dim + cond_dim, (C,H,W), up_width, up_depth)
        self.cbam = CBAM(C, cbam_reduction, cbam_kernel) if enable_cbam else nn.Identity()
        self.tok = SinCosTime(time_dim)
        self.fusion = FiLM2d(64, C) if fusion == "film" else nn.Identity()

        self.unet = UNetModel(
            image_size=H, in_channels=C, out_channels=C,
            model_channels=unet_channels, num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attn_resolutions),
            dropout=dropout, num_classes=None
        )
        self.down = Downsampler((C,H,W), y_dim)

    def forward(self, x, t, condition):
        # x:(B,y_dim), condition:(B,cond_dim), t:(B,)
        h = torch.cat([x, condition], dim=1)
        img = self.ups(h)                   # (B,C,H,W)
        img = self.cbam(img)
        f = self.tok(t)                     # (B,64)
        if isinstance(self.fusion, FiLM2d): img = self.fusion(img, f)
        # fmtorch UNetModel 习惯: forward(t, x, y=None)
        v_img = self.unet(t, img, y=None)   # (B,C,H,W)
        v = self.down(v_img)                # (B,y_dim)
        return v

# ======================================================================================
#                            Helper Modules
# ======================================================================================
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class ResidualFC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = Mish()

    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(out + x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        hidden = max(channels // ratio, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False), Mish(),
            nn.Conv2d(hidden, channels, 1, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        att = self.fc(self.avg(x) + self.max(x))
        return x * att

class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k//2, bias=False)

    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))
        return x * att

class UNetWithAttention(nn.Module):
    def __init__(self, unet_base, in_channels):
        super().__init__()
        self.unet = unet_base
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, t, x, y=None):
        feat = self.unet(t, x, y)
        feat = self.ca(feat)
        feat = self.sa(feat)
        return feat

# ======================================================================================
#                            ImgUNetFlowModel (from notebook)
# ======================================================================================
class UpSampler(nn.Module):
    def __init__(self, in_shape=(604,), out_shape=(1,28,28), width=1024, depth=5):
        super().__init__()
        in_dim, out_dim = np.prod(in_shape), np.prod(out_shape)
        layers = [nn.Linear(in_dim, width), Mish()]
        for _ in range(depth-2):
            layers.append(ResidualFC(width))
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)
        self.out_shape = out_shape

    def forward(self, x):
        x = x.flatten(1)
        x = self.net(x)
        x = self.norm(x)
        return x.view(x.size(0), *self.out_shape)

class DownSampler(nn.Module):
    def __init__(self, in_shape=(1,28,28), out_shape=(604,), width=1024, depth=5):
        super().__init__()
        in_dim, out_dim = np.prod(in_shape), np.prod(out_shape)
        layers = [nn.Linear(in_dim, width), Mish()]
        for _ in range(depth-2):
            layers.append(ResidualFC(width))
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_shape = out_shape

    def forward(self, x):
        x = x.flatten(1)
        x = self.net(x)
        return x.view(x.size(0), *self.out_shape)

class ImgUNetFlowModel(nn.Module):
    """
    U-Net model that process trajectory data as an image.
    It includes an upsampler to convert trajectory to image, a UNet, and a downsampler.
    Handles both conditional (cond_dim > 0) and unconditional (cond_dim == 0) cases.
    """
    def __init__(self, cfg, y_dim, cond_dim):
        super().__init__()
        self.y_dim = y_dim
        self.cond_dim = cond_dim
        img_shape = (cfg.img_shape.C, cfg.img_shape.H, cfg.img_shape.W)

        self.upsampler = UpSampler(
            in_shape=(y_dim,),
            out_shape=img_shape,
            width=cfg.width,
            depth=cfg.depth
        )
        self.downsampler = DownSampler(
            in_shape=img_shape,
            out_shape=(y_dim,),
            width=cfg.width,
            depth=cfg.depth
        )
        
        unet_in_channels = img_shape[0] + (cond_dim if self.cond_dim > 0 else 0)
        
        unet_base = UNetModel(
            image_size=img_shape[-1],
            in_channels=unet_in_channels,
            model_channels=cfg.unet.model_channels,
            out_channels=img_shape[0],
            num_res_blocks=cfg.unet.num_res_blocks,
            attention_resolutions=cfg.unet.attention_resolutions,
            channel_mult=cfg.unet.channel_mult,
            num_classes=cfg.unet.num_classes if self.cond_dim > 0 else None,
        )
        self.unet = UNetWithAttention(unet_base, in_channels=img_shape[0])


    def forward(self, y, t, condition=None):
        # 1. Upsample y to image space
        img_y = self.upsampler(y) # (B, C, H, W)

        # 2. Handle conditioning
        if self.cond_dim > 0:
            assert condition is not None, "Condition must be provided for conditional model"
            # Option 1: Concat condition to image channels
            cond_img = condition.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, img_y.shape[2], img_y.shape[3])
            img_in = torch.cat([img_y, cond_img], dim=1)
            # Option 2: Use as class label (if num_classes is set in UNet)
            # Here we assume Option 1 is used, as per the channel setup.
            unet_cond = None 
        else:
            img_in = img_y
            unet_cond = None

        # 3. UNet processes the combined image
        v_img = self.unet(t, img_in, y=unet_cond)

        # 4. Downsample back to trajectory space
        v_y = self.downsampler(v_img)
        return v_y

# ======================================================================================
#                            FlowUNet_CBAM (from single-step notebook)
# ======================================================================================
class MHSEChannelAttention(nn.Module):
    """
    Multi-Head Squeeze-Excitation Channel Attention.
    Splits channels into heads, applies SE to each, and fuses the results.
    """
    def __init__(self, channels, reduction=16, heads=8, fuse='concat'):
        super().__init__()
        assert channels % heads == 0, "channels must be divisible by heads"
        self.channels = channels
        self.heads = heads
        self.fuse = fuse
        self.subc = channels // heads
        
        self.se_blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.subc, self.subc // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(self.subc // reduction, self.subc, bias=False),
                nn.Sigmoid()
            ) for _ in range(heads)
        ])
        if fuse == 'concat':
            self.fuse_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        xs = x.view(B, self.heads, self.subc, H, W)
        weights = [se(xs[:, i]) for i, se in enumerate(self.se_blocks)]

        if self.fuse == 'mean':
            w = torch.stack(weights, dim=0).mean(0).view(B, self.subc, 1, 1)
            out = xs * w.unsqueeze(1)
            return out.view(B, C, H, W)
        else: # 'concat'
            out_heads = [xs[:, i] * w.view(B, self.subc, 1, 1) for i, w in enumerate(weights)]
            out = torch.cat(out_heads, dim=1)
            return self.fuse_conv(out)

class CBAMWithMHSE(nn.Module):
    """CBAM with Multi-Head SE Channel Attention"""
    def __init__(self, channels, reduction=16, heads=4, kernel_size=7):
        super().__init__()
        self.ca = MHSEChannelAttention(channels, reduction, heads, fuse='concat')
        padding = (kernel_size - 1) // 2
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.ca(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.sa(torch.cat([avg_out, max_out], dim=1))
        return x

class FlowUNet_CBAM(nn.Module):
    def __init__(self, cfg, y_dim, cond_dim):
        super().__init__()
        total_dim = y_dim + cond_dim
        img_shape = (cfg.img_shape.C, cfg.img_shape.H, cfg.img_shape.W)

        self.upsampler = UpSampler(
            in_shape=(total_dim,),
            out_shape=img_shape,
            width=cfg.width,
            depth=cfg.depth
        )
        self.downsampler = DownSampler(
            in_shape=img_shape,
            out_shape=(y_dim,),
            width=cfg.width,
            depth=cfg.depth
        )
        self.unet = UNetModel(
            image_size=img_shape[-1],
            in_channels=img_shape[0],
            model_channels=cfg.unet.model_channels,
            out_channels=img_shape[0],
            num_res_blocks=cfg.unet.num_res_blocks,
            attention_resolutions=cfg.unet.attention_resolutions,
            channel_mult=cfg.unet.channel_mult,
            num_classes=None, # CBAM model is unconditional in its UNet part
        )
        self.attention = CBAMWithMHSE(
            channels=img_shape[0],
            reduction=cfg.cbam.reduction,
            heads=cfg.cbam.heads,
            kernel_size=cfg.cbam.kernel_size
        )

    def forward(self, y, t, condition):
        # 1. Concatenate condition and target at the input level
        inp = torch.cat([y, condition], dim=1)
        
        # 2. Upsample to image space
        img = self.upsampler(inp)

        # 3. Apply CBAM attention
        img_att = self.attention(img)
        
        # 4. UNet processes the image
        v_img = self.unet(t, img_att)
        
        # 5. Downsample back to trajectory space
        v_y = self.downsampler(v_img)
        return v_y


def build_model(cfg, **kwargs):
    if cfg.model.name == "simple_mlp":
        return SimpleMLP(cfg.model, **kwargs)
    elif cfg.model.name == "img_unet":
        return ImgUNetFlowModel(cfg.model, kwargs['y_dim'], kwargs['cond_dim'])
    elif cfg.model.name == "flow_unet_cbam":
        return FlowUNet_CBAM(cfg.model, kwargs['y_dim'], kwargs['cond_dim'])
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")
