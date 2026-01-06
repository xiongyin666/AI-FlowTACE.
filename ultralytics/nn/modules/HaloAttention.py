# 论文名称：Scaling Local Self-Attention for Parameter Efficient Visual Backbones
# 博客地址：https://blog.csdn.net/weixin_43694096/article/details/138729980
# 论文地址：https://openaccess.thecvf.com/content/CVPR2021/papers/Vaswani_Scaling_Local_Self-Attention_for_Parameter_Efficient_Visual_Backbones_CVPR_2021_paper.pdf
#
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat


def to(x):
    return {"device": x.device, "dtype": x.dtype}


def pair(x):
    return (x, x) if not isinstance(x, tuple) else x


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim=2)
    flat_x = rearrange(x, "b l c -> b (l c)")
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x


def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum("b x y d, r d -> b x y r", q, rel_k)
    logits = rearrange(logits, "b x y r -> (b x) y r")
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim=2, k=r)
    return logits


class RelPosEmb(nn.Module):
    def __init__(self, block_size, rel_size, dim_head):
        super().__init__()
        height = width = rel_size
        scale = dim_head**-0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, "b (x y) c -> b x y c", x=block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, "b x i y j-> b (x y) (i j)")

        q = rearrange(q, "b x y d -> b y x d")
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, "b x i y j -> b (y x) (j i)")
        return rel_logits_w + rel_logits_h


class HaloAttention(nn.Module):
    def __init__(self, dim, block_size, halo_size, dim_head=64, heads=8):
        super().__init__()
        assert halo_size > 0, "halo size must be greater than 0"

        self.dim = dim
        self.heads = heads
        self.scale = dim_head**-0.5
        self.block_size = block_size
        self.halo_size = halo_size
        inner_dim = dim_head * heads

        self.rel_pos_emb = RelPosEmb(
            block_size=block_size,
            rel_size=block_size + (halo_size * 2),
            dim_head=dim_head,
        )
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def pad_to_block_size(self, x):
        _, _, h, w = x.shape
        h_pad = (self.block_size - h % self.block_size) % self.block_size
        w_pad = (self.block_size - w % self.block_size) % self.block_size
        x_padded = F.pad(x, (0, w_pad, 0, h_pad))  # 在底部和右侧进行填充
        return x_padded, h, w  # 返回填充后的张量和原始高度、宽度

    def forward(self, x):
        # 获取原始的高度和宽度，并对输入进行填充
        x, original_h, original_w = self.pad_to_block_size(x)

        b, c, h, w = x.shape
        block, halo, heads, device = (
            self.block_size,
            self.halo_size,
            self.heads,
            x.device,
        )
        assert (
            h % block == 0 and w % block == 0
        ), "fmap dimensions must be divisible by the block size"
        assert (
            c == self.dim
        ), f"channels for input ({c}) does not equal to the correct dimension ({self.dim})"

        # get block neighborhoods, and prepare a halo-ed version (blocks with padding) for deriving key values

        q_inp = rearrange(
            x, "b c (h p1) (w p2) -> (b h w) (p1 p2) c", p1=block, p2=block
        )

        kv_inp = F.unfold(x, kernel_size=block + halo * 2, stride=block, padding=halo)
        kv_inp = rearrange(kv_inp, "b (c j) i -> (b i) j c", c=c)

        # derive queries, keys, values

        q = self.to_q(q_inp)
        k, v = self.to_kv(kv_inp).chunk(2, dim=-1)

        # split heads

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=heads), (q, k, v)
        )

        # scale

        q *= self.scale

        # attention

        sim = einsum("b i d, b j d -> b i j", q, k)

        # add relative positional bias

        sim += self.rel_pos_emb(q)

        # mask out padding (in the paper, they claim to not need masks, but what about padding?)

        mask = torch.ones(1, 1, h, w, device=device)
        mask = F.unfold(
            mask, kernel_size=block + (halo * 2), stride=block, padding=halo
        )
        mask = repeat(mask, "() j i -> (b i h) () j", b=b, h=heads)
        mask = mask.bool()

        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate

        out = einsum("b i j, b j d -> b i d", attn, v)

        # merge and combine heads

        out = rearrange(out, "(b h) n d -> b n (h d)", h=heads)
        out = self.to_out(out)

        # merge blocks back to original feature map

        # 合并块回原始特征图
        out = rearrange(
            out,
            "(b nh nw) (p1 p2) c -> b c (nh p1) (nw p2)",
            b=b,
            nh=(h // block),
            nw=(w // block),
            p1=block,
            p2=block,
        )

        # 将输出裁剪回原始的高度和宽度
        out = out[:, :, :original_h, :original_w]

        return out


if __name__ == "__main__":
    input = torch.rand(3, 32, 64, 64).cuda()
    model = HaloAttention(
        dim=32,
        block_size=2,
        halo_size=1,
    ).cuda()
    output = model(input)
    print(input.size(), output.size())

