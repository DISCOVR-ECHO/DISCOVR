from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.utils.checkpoint as checkpoint
from discovr.models.TubeViT.tubevit.model import SparseTubesTokenizer
from discovr.models.TubeViT.tubevit.positional_encoding import get_3d_sincos_pos_embed


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


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
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return  torch.tensor(sinusoid_table,dtype=torch.float, requires_grad=False).unsqueeze(0) 


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 fc_drop_rate=0., 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False, 
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_checkpoint=False,
                 use_mean_pooling=True,
                 use_cls=True,
                 tokenizer_type='default',
                 pretrained_cfg=None, 
                 pretrained_cfg_overlay=None,
                 decoder_depth=None,
                 decoder_num_classes=None,
                 target_type=None,
                 loss_func=None,
                 num_prototypes=None,
                 sinkhorn_iterations=None,
                 eps=None,
                 skip_dino_loss=False,
                 use_dino_crop=False
                 ):
        super().__init__()
        print(f"\nDebug VisionTransformer init:")
        print(f"tokenizer_type: {tokenizer_type}")
        print(f"num_frames: {all_frames}")
        print(f"tubelet_size: {tubelet_size}")
        print(f"img_size: {img_size}")
        print(f"patch_size: {patch_size}")
        print(f"in_chans: {in_chans}")
        print(f"embed_dim: {embed_dim}")
        print(f"depth: {depth}")
        print(f"num_heads: {num_heads}")
        print(f"mlp_ratio: {mlp_ratio}")
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.num_frames = all_frames
        self.tokenizer_type = tokenizer_type
        
        if tokenizer_type == 'default':
            self.patch_embed = PatchEmbed(
                img_size=img_size, 
                patch_size=patch_size, 
                in_chans=in_chans, 
                embed_dim=embed_dim,
                num_frames=all_frames, 
                tubelet_size=self.tubelet_size
            )
            print(f"Num patches after PatchEmbed: {self.patch_embed.num_patches}")
            num_patches = self.patch_embed.num_patches
            self.patch_embed.patch_size = patch_size
        elif tokenizer_type == 'sparse_tube':  # sparse_tube
            kernel_sizes = (
                (8, 8, 8),
                (16, 4, 4),
                (4, 12, 12),
                (1, 16, 16),
            )
            strides = (
                (16, 32, 32),
                (6, 32, 32),
                (16, 32, 32),
                (32, 16, 16),
            )
            offsets = (
                (0, 0, 0),
                (4, 8, 8),
                (0, 16, 16),
                (0, 0, 0),
            )
            
            self.patch_embed = SparseTubesTokenizer(
                hidden_dim=embed_dim,
                kernel_sizes=kernel_sizes,
                strides=strides,
                offsets=offsets
            )
            
            self.patch_embed.video_shape = (3, all_frames, img_size, img_size)
            self.patch_embed.tokenizer_type = 'sparse_tube'
            
            tube_token_counts = []
            total_tokens = 0
            
            for kernel, stride, offset in zip(self.patch_embed.kernel_sizes, 
                                             self.patch_embed.strides, 
                                             self.patch_embed.offsets):
                t_tokens = max(1, (all_frames - offset[0] - kernel[0] + 1 + stride[0] - 1) // stride[0])
                h_tokens = max(1, (img_size - offset[1] - kernel[1] + 1 + stride[1] - 1) // stride[1])
                w_tokens = max(1, (img_size - offset[2] - kernel[2] + 1 + stride[2] - 1) // stride[2])
                
                token_count = t_tokens * h_tokens * w_tokens
                tube_token_counts.append(token_count)
                total_tokens += token_count
            
            num_patches = total_tokens
            self.patch_embed.num_patches = num_patches
            print(f"INFO: Calculated {num_patches} tokens for sparse tube positional encoding")
        
        self.use_checkpoint = use_checkpoint
        self.use_cls = use_cls

        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=.02)

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + (1 if use_cls else 0), embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
        else:
            if tokenizer_type == 'sparse_tube':
                pos_embeds = []
                
                if use_cls:
                    cls_pos_embed = np.zeros([1, embed_dim], dtype=np.float32)
                    pos_embeds.append(cls_pos_embed)
                
                for i, (kernel, stride, offset) in enumerate(zip(self.patch_embed.kernel_sizes, 
                                                               self.patch_embed.strides, 
                                                               self.patch_embed.offsets)):
                    t_size = max(1, (all_frames - offset[0] - kernel[0] + 1 + stride[0] - 1) // stride[0])
                    h_size = max(1, (img_size - offset[1] - kernel[1] + 1 + stride[1] - 1) // stride[1])
                    w_size = max(1, (img_size - offset[2] - kernel[2] + 1 + stride[2] - 1) // stride[2])
                    
                    tube_shape = (t_size, h_size, w_size)
                    
                    tube_pos_embed = get_3d_sincos_pos_embed(
                        embed_dim, 
                        tube_shape,
                        stride=stride, 
                        offset=offset, 
                        kernel_size=kernel,
                        cls_token=False
                    )
                    
                    pos_embeds.append(tube_pos_embed)
                    
                pos_embed = np.concatenate(pos_embeds, axis=0)
                
            else:
                pos_embed = get_sinusoid_encoding_table(num_patches + (1 if use_cls else 0), embed_dim)
                
            self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float().unsqueeze(0), requires_grad=False)

        # Add local position embedding for DINO crop
        if use_dino_crop and not use_learnable_pos_emb:
            local_num_patches = (96 // patch_size) * (96 // patch_size)
            local_pos_embed = get_sinusoid_encoding_table(local_num_patches, embed_dim)
            if use_cls:
                cls_pos_embed = torch.zeros(1, 1, embed_dim)
                local_pos_embed = torch.cat([cls_pos_embed, local_pos_embed], dim=1)
            self.local_pos_embed = nn.Parameter(local_pos_embed, requires_grad=False)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B, C, T, H, W = x.shape
        x = self.patch_embed(x)
        B, N, C = x.size()

        if self.use_cls:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:   
            for blk in self.blocks:
                x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(self.fc_dropout(x))
        return x


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    tokenizer_type = kwargs.pop('tokenizer_type', 'default') if 'tokenizer_type' in kwargs else 'default'
    
    model = VisionTransformer(
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        tokenizer_type=tokenizer_type,
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_512(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_huge_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
