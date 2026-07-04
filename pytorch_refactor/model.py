import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import collections.abc
from itertools import repeat

# Optional DeepSpeed MoE
try:
    from deepspeed.moe.layer import MoE as _DeepSpeedMoE  # type: ignore
except Exception:  # pragma: no cover
    _DeepSpeedMoE = None


def _checkpoint(fn, *args):
    try:
        # PyTorch 2.9+ requires explicit use_reentrant.
        return checkpoint.checkpoint(fn, *args, use_reentrant=False)
    except TypeError:  # pragma: no cover
        # Older PyTorch versions.
        return checkpoint.checkpoint(fn, *args)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_moe=False, num_experts=1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_moe = bool(use_moe)

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        if self.use_moe:
            if _DeepSpeedMoE is None:
                raise ImportError(
                    "DeepSpeed is required for MoE layers, but 'deepspeed' is not installed. "
                    "Install deepspeed or disable MoE."
                )
            self.mlp = _DeepSpeedMoE(
                hidden_size=dim,
                expert=Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop),
                num_experts=num_experts,
                k=1,
                min_capacity=0,
            )
        else:
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)

        # aux_loss = 0
        aux_loss = x.new_zeros(())  # scalar tensor on correct device/dtype
        if self.use_moe:
            # DeepSpeed MoE forward typically returns (out, gate_loss, expert_counts)
            x_mlp, aux_loss, _ = self.mlp(self.norm2(x))
        else:
            x_mlp = self.mlp(self.norm2(x))

        x = x + self.drop_path(x_mlp)

        return x, aux_loss

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class MultiModalSwinTransformerV2MoE(nn.Module):
    def __init__(
        self,
        img_size=192,
        patch_size=4,
        modal_in_chans=None,
        num_classes=0,
        embed_dim=96,
        depths=None,
        num_heads=None,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        moe_config=None,
    ):
        super().__init__()

        if modal_in_chans is None:
            raise ValueError("modal_in_chans must be provided for MultiModalSwinTransformerV2MoE")
        self.modal_in_chans = [int(c) for c in modal_in_chans]
        self.modal_num = len(self.modal_in_chans)
        if self.modal_num < 2:
            raise ValueError(f"modal_in_chans must have length >= 2, got {self.modal_num}")

        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embeds = nn.ModuleList(
            [
                PatchEmbed(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    norm_layer=norm_layer if self.patch_norm else None,
                )
                for in_chans in self.modal_in_chans
            ]
        )

        num_patches = self.patch_embeds[0].num_patches
        patches_resolution = self.patch_embeds[0].patches_resolution
        for pe in self.patch_embeds[1:]:
            if pe.num_patches != num_patches or pe.patches_resolution != patches_resolution:
                raise ValueError("All modalities must share the same img_size/patch_size so num_patches match")
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            use_moe_layer = False
            num_experts = 1
            if moe_config and i_layer in moe_config.get("moe_stages", []):
                use_moe_layer = True
                num_experts = moe_config.get("num_experts", 1)

            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(patches_resolution[0] // (2**i_layer), patches_resolution[1] // (2**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                use_moe=use_moe_layer,
                num_experts=num_experts,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patch_embed(self, modal_idx: int, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embeds[modal_idx](x)


class SwinTransformerV2MoE(nn.Module):
    def __init__(self, img_size=192, patch_size=4, in_chans=3, num_classes=0,
                 embed_dim=96, depths=None, num_heads=None,
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, moe_config=None):
        super().__init__()

        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # Determine if this layer should use MoE
            # For simplicity, let's say we use MoE in the last 2 stages or based on config
            use_moe_layer = False
            num_experts = 1
            if moe_config and i_layer in moe_config.get('moe_stages', []):
                use_moe_layer = True
                num_experts = moe_config.get('num_experts', 1)

            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               use_moe=use_moe_layer,
                               num_experts=num_experts)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        total_aux_loss = x.new_zeros(())
        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss = total_aux_loss + aux_loss

        if self.norm is not None:
            x = self.norm(x)
        return x, total_aux_loss

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_moe=False, num_experts=1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 use_moe=use_moe and (i % 2 == 1), # Apply MoE on every other block or as configured
                                 num_experts=num_experts)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        total_aux_loss = x.new_zeros(())
        for blk in self.blocks:
            if self.use_checkpoint:
                x, aux_loss = _checkpoint(blk, x)
            else:
                x, aux_loss = blk(x)
            total_aux_loss = total_aux_loss + aux_loss

        if self.downsample is not None:
            x = self.downsample(x)
        return x, total_aux_loss

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x

class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride=32):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3,
                kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.encoder.patch_embed.in_chans
        self.patch_size = self.encoder.patch_embed.patch_size[0]

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.encoder.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):
        """Forward SimMIM.

        Args:
            x: [B, 3, H, W]
            mask: patch-level mask [B, H_p, W_p] where H_p = H/patch_size
        """
        z = self.encoder.patch_embed(x)  # [B, L, C_embed]

        B, L, C_embed = z.shape
        H_p, W_p = self.encoder.patches_resolution

        if mask.dim() != 3:
            raise ValueError(f"Expected mask with shape [B, H_p, W_p], got {tuple(mask.shape)}")
        if mask.shape[1] != H_p or mask.shape[2] != W_p:
            raise ValueError(f"Mask shape {tuple(mask.shape)} must match patches_resolution {(H_p, W_p)}")

        mask_tokens = self.mask_token.expand(B, L, -1)
        mask_patches = mask.flatten(1).to(dtype=torch.bool)  # [B, L]

        w = mask_patches.unsqueeze(-1).type_as(z)
        z = z * (1 - w) + mask_tokens * w

        if self.encoder.ape:
            z = z + self.encoder.absolute_pos_embed
        z = self.encoder.pos_drop(z)

        total_aux_loss = z.new_zeros(())
        for layer in self.encoder.layers:
            z, aux_loss = layer(z)
            total_aux_loss = total_aux_loss + aux_loss

        z = self.encoder.norm(z)

        # After encoder stages, tokens correspond to a lower-res grid due to PatchMerging.
        # For Swin: each stage (except last) downsamples by 2.
        num_down = max(len(self.encoder.layers) - 1, 0)
        H_feat = H_p // (2 ** num_down)
        W_feat = W_p // (2 ** num_down)

        C_enc = z.shape[-1]
        z = z.transpose(1, 2).contiguous().view(B, C_enc, H_feat, W_feat)
        x_rec = self.decoder(z)

        # Convert patch mask to pixel mask for reconstruction loss.
        # SimMIM loss is typically applied on masked pixels; our decoder outputs full-res.
        pixel_mask = mask.unsqueeze(1).to(dtype=x.dtype)
        pixel_mask = pixel_mask.repeat_interleave(self.patch_size, dim=2).repeat_interleave(self.patch_size, dim=3)

        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * pixel_mask).sum() / (pixel_mask.sum() + 1e-5) / self.in_chans

        return loss, x_rec, total_aux_loss


class MultiModalSimMIM(nn.Module):
    def __init__(self, encoder: MultiModalSwinTransformerV2MoE, encoder_stride: int = 32, modal_in_chans=None):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.modal_in_chans = [int(c) for c in (modal_in_chans or encoder.modal_in_chans)]
        if len(self.modal_in_chans) != encoder.modal_num:
            raise ValueError("modal_in_chans length must match encoder.modal_num")

        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.encoder.num_features,
                        out_channels=self.encoder_stride**2 * in_chans,
                        kernel_size=1,
                    ),
                    nn.PixelShuffle(self.encoder_stride),
                )
                for in_chans in self.modal_in_chans
            ]
        )

        self.patch_size = self.encoder.patch_embeds[0].patch_size[0]

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.encoder.embed_dim))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)

    def forward(self, *inputs):
        if len(inputs) % 2 != 0:
            raise ValueError(f"Expected even number of inputs (x0,mask0,x1,mask1,...), got {len(inputs)}")

        modal_num = len(inputs) // 2
        if modal_num != self.encoder.modal_num:
            raise ValueError(f"Expected {self.encoder.modal_num} modalities, got {modal_num}")

        xs = [inputs[i * 2] for i in range(modal_num)]
        masks = [inputs[i * 2 + 1] for i in range(modal_num)]

        batch = xs[0].shape[0]
        for i, x in enumerate(xs):
            if x.dim() != 4:
                raise ValueError(f"Expected x[{i}] shape [B,C,H,W], got {tuple(x.shape)}")
            if x.shape[0] != batch:
                raise ValueError("All modalities must have the same batch size")
            if x.shape[1] != self.modal_in_chans[i]:
                raise ValueError(f"Expected x[{i}] to have {self.modal_in_chans[i]} channels, got {x.shape[1]}")

        H_p, W_p = self.encoder.patches_resolution
        for i, m in enumerate(masks):
            if m.dim() != 3:
                raise ValueError(f"Expected mask[{i}] shape [B,H_p,W_p], got {tuple(m.shape)}")
            if m.shape[0] != batch:
                raise ValueError("All modalities must have the same batch size")
            if (m.shape[1], m.shape[2]) != (H_p, W_p):
                raise ValueError(f"mask[{i}] shape must be (B,{H_p},{W_p}), got {tuple(m.shape)}")

        # Patch-embed each modality (supports different channel counts), then concat on batch dim.
        z_list = [self.encoder.patch_embed(i, xs[i]) for i in range(modal_num)]  # each [B,L,C]
        z = torch.cat(z_list, dim=0)  # [B*M, L, C]
        mask_all = torch.cat(masks, dim=0)  # [B*M, H_p, W_p]

        # Apply SimMIM mask token.
        B_total, L, _ = z.shape
        mask_tokens = self.mask_token.expand(B_total, L, -1)
        mask_patches = mask_all.flatten(1).to(dtype=torch.bool)  # [B*M, L]
        w = mask_patches.unsqueeze(-1).type_as(z)
        z = z * (1 - w) + mask_tokens * w

        if self.encoder.ape:
            z = z + self.encoder.absolute_pos_embed
        z = self.encoder.pos_drop(z)

        total_aux_loss = z.new_zeros(())
        for layer in self.encoder.layers:
            z, aux_loss = layer(z)
            total_aux_loss = total_aux_loss + aux_loss

        z = self.encoder.norm(z)

        # Swin stages downsample by 2 at the end of each stage except the last.
        num_down = max(len(self.encoder.layers) - 1, 0)
        H_feat = H_p // (2**num_down)
        W_feat = W_p // (2**num_down)

        # Split back to per-modality batches.
        z_splits = z.split(batch, dim=0)

        losses = []
        recons = []
        for i in range(modal_num):
            z_i = z_splits[i]  # [B, L, C_enc]
            C_enc = z_i.shape[-1]
            z_i = z_i.transpose(1, 2).contiguous().view(batch, C_enc, H_feat, W_feat)
            x_rec = self.decoders[i](z_i)
            recons.append(x_rec)

            x = xs[i]
            m = masks[i].unsqueeze(1).to(dtype=x.dtype)
            m = m.repeat_interleave(self.patch_size, dim=2).repeat_interleave(self.patch_size, dim=3)

            loss_recon = F.l1_loss(x, x_rec, reduction="none")
            loss_i = (loss_recon * m).sum() / (m.sum() + 1e-5) / x.shape[1]
            losses.append(loss_i)

        loss = sum(losses)
        return loss, recons, total_aux_loss

