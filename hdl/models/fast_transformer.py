# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/models/fast_transformer.py
# 说明：神经网络模型定义与注册表
# 模块功能：高效 Transformer（linear / FAST attention）骨干，用全局查询-键 token 代替成对注意力打分做序列建模，输出每个形状 (batch, 序列长度, 词表) 的 logits。
import torch
# import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, reduce
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding

# helper functions


def exists(val):
    """判断对象是否存在（val is not None），返回 bool。"""
    return val is not None


def default(val, d):
    """取值：val 存在则返回 val，否则返回默认值 d。"""
    return val if exists(val) else d


# helper classes
class PreNorm(nn.Module):
    """前置归一化（pre-normalization）包装器：先对输入做 LayerNorm，再调用被包装的子模块 fn。"""
    def __init__(self, dim, fn):
        """dim 为 LayerNorm 的归一化维（即特征维），fn 为被包装的子模块；建 self.norm=nn.LayerNorm(dim) 与 self.fn，供 forward 先归一化再调 fn。"""
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """x: (batch, 序列长度, dim)，先 LayerNorm 再送入 fn，返回同形状结果。"""
        x = self.norm(x)
        return self.fn(x, **kwargs)

# blocks


def FeedForward(dim, mult=4):
    """前馈子层（feed-forward）：dim -> dim*mult 的 Linear + GELU 再降回 dim。"""
    return nn.Sequential(
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Linear(dim * mult, dim)
    )


class FastAttention(nn.Module):
    """线性注意力（linear / FAST attention）子层：不显式计算成对注意力矩阵，
    而是先用掩码 softmax（masked attention）把查询、键各压成一个全局 token（global query/key token），
    再用逐元素乘积近似注意力权重，最后对值做线性变换并加查询残差。
    forward 输入 x (batch, 序列长度, dim)，返回同形状张量。"""
    def __init__(
        self,
        dim,
        *,
        heads=8,
        dim_head=64,
        max_seq_len=None,
        pos_emb=None
    ):
        """dim：输入/输出特征维；heads：注意力头数；dim_head：每头特征维（内部维 inner_dim = heads * dim_head）；
        max_seq_len：使用旋转位置编码（rotary positional embedding）时必需的最大序列长度；
        pos_emb：旋转位置编码模块，None 表示不做位置编码。"""
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        # 一次性把输入投影成 q/k/v 三个 inner_dim 分支（无 bias）
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # rotary positional embedding

        assert not (exists(pos_emb) and not exists(max_seq_len)), \
            'max_seq_len must be passed in if to use rotary positional embeddings'

        self.pos_emb = pos_emb
        self.max_seq_len = max_seq_len

        # if using relative positional encoding, make sure to reduce pairs of 
        # consecutive feature dimension before doing projection to attention logits

        kv_attn_proj_divisor = 1 if not exists(pos_emb) else 2

        # for projecting queries to query attention logits
        # 查询侧打分投影：dim_head -> 1，得到每个位置的查询注意力标量
        self.to_q_attn_logits = nn.Linear(dim_head, 1, bias=False) 
        # 键侧打分投影：dim_head（旋转编码下已减半）-> 1，得到每个位置的键注意力标量
        self.to_k_attn_logits = nn.Linear(
            dim_head // kv_attn_proj_divisor,
            1,
            bias=False
        )  # for projecting keys to key attention logits

        # final transformation of values to "r" as in the paper

        # 值侧最终投影：被全局键 token 偏置后的 u（旋转编码下维度减半）映回 dim_head
        self.to_r = nn.Linear(dim_head // kv_attn_proj_divisor, dim_head)

        # 多头结果拼接（heads * dim_head）后投影回模型维 dim
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask=None):
        """x: (batch, 序列长度 n, dim)；mask: (batch, n) 的 bool 张量，标记参与注意力的有效位置。
        返回 (batch, n, dim)。"""
        n, device, h, use_rotary_emb = x.shape[1], x.device, self.heads, exists(self.pos_emb)

        # 投影并拆成 q/k/v，再把内部维按头重排为 (batch, 头数, n, dim_head)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # 无效位置的打分置为该数据类型最小值，softmax 后权重趋近 0
        mask_value = -torch.finfo(x.dtype).max
        mask = rearrange(mask, 'b n -> b () n')

        # if relative positional encoding is needed

        if use_rotary_emb:
            # 旋转位置编码（rotary）：按位置生成频率并作用到 q/k/v 上
            freqs = self.pos_emb(torch.arange(self.max_seq_len, device=device), cache_key=self.max_seq_len)
            freqs = rearrange(freqs[:n], 'n d -> () () n d')
            q_aggr, k_aggr, v_aggr = map(lambda t: apply_rotary_emb(freqs, t), (q, k, v))
        else:
            q_aggr, k_aggr, v_aggr = q, k, v

        # calculate query attention logits

        # 查询打分投影 + 缩放，掩码位置填极小值后 softmax，得到查询在序列上的分布
        q_attn_logits = rearrange(self.to_q_attn_logits(q), 'b h n () -> b h n') * self.scale
        q_attn_logits = q_attn_logits.masked_fill(~mask, mask_value)
        q_attn = q_attn_logits.softmax(dim=-1)

        # calculate global query token

        # 用查询分布对旋转编码后的 q 加权求和，得到每头的全局查询 token
        global_q = einsum('b h n, b h n d -> b h d', q_attn, q_aggr)
        global_q = rearrange(global_q, 'b h d -> b h () d')

        # bias keys with global query token

        # 用全局查询 token 逐元素偏置键，替代成对注意力打分
        k = k * global_q

        # if using rotary embeddings, do an inner product between adjacent pairs in the feature dimension

        if use_rotary_emb:
            # 相邻两个特征维配对求和，把每头维度减半（配合旋转编码的成对结构）
            k = reduce(k, 'b h n (d r) -> b h n d', 'sum', r=2)

        # now calculate key attention logits

        # 键打分投影 + 缩放 + 掩码 softmax，得到键在序列上的分布
        k_attn_logits = rearrange(self.to_k_attn_logits(k), 'b h n () -> b h n') * self.scale
        k_attn_logits = k_attn_logits.masked_fill(~mask, mask_value)
        k_attn = k_attn_logits.softmax(dim=-1)

        # calculate global key token

        # 同样用键分布对旋转编码后的 k/v 加权求和，聚出每头的全局键 token
        global_k = einsum('b h n, b h n d -> b h d', k_attn, k_aggr)
        global_k = rearrange(global_k, 'b h d -> b h () d')

        # bias the values

        # 全局键 token 偏置值，得到线性注意力的加权输出 u
        u = v_aggr * global_k

        # if using rotary embeddings, do an inner product between adjacent pairs in the feature dimension

        if use_rotary_emb:
            # 同键侧：相邻特征维配对降维，保证 to_r 输入维度一致
            u = reduce(u, 'b h n (d r) -> b h n d', 'sum', r=2)

        # transformation step

        # 线性投影回 dim_head
        r = self.to_r(u)

        # paper then says to add the queries as a residual

        # 查询作为残差加回（注意力子层的残差连接）
        r = r + q

        # combine heads

        # 各头拼回 (batch, n, heads*dim_head) 后投影到 dim
        r = rearrange(r, 'b h n d -> b n (h d)')
        return self.to_out(r)


# main class
class FastTransformer(nn.Module):
    """高效 Transformer（FAST Transformer）序列模型：token 嵌入 + depth 层「线性注意力 + 前馈」块（均带前置归一化与残差），
    对离散 token 序列建模。forward 输入 (batch, 序列长度) 的 token id 整数张量，输出 (batch, 序列长度, num_tokens) 的词表 logits。"""
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        heads=8,
        dim_head=64,
        ff_mult=4,
        absolute_pos_emb=False
    ):
        """num_tokens：词表大小（输入 token 数与输出 logits 维）；dim：模型隐藏维；depth：Transformer 块数；
        max_seq_len：最大序列长度；heads：注意力头数；dim_head：每头维度；ff_mult：前馈层扩展倍数；
        absolute_pos_emb：True 用绝对位置嵌入，False 用共享的旋转位置编码。"""
        super().__init__()
        # 输入 token 嵌入
        self.token_emb = nn.Embedding(num_tokens, dim)

        # positional embeddings

        # 绝对位置嵌入（可选）：按序列位置查表
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if absolute_pos_emb else None

        # 非绝对位置编码时，各层共享一个旋转位置编码模块，占每头维度的一半
        layer_pos_emb = None
        if not absolute_pos_emb:
            assert (dim_head % 4) == 0, 'dimension of the head must be divisible by 4 to use rotary embeddings'
            layer_pos_emb = RotaryEmbedding(dim_head // 2)

        # layers

        # depth 个块，每块为 [PreNorm(线性注意力), PreNorm(前馈)]，堆叠顺序在 forward 中按残差依次串联
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            attn = FastAttention(
                dim,
                dim_head=dim_head,
                heads=heads,
                pos_emb=layer_pos_emb,
                max_seq_len=max_seq_len
            )
            ff = FeedForward(dim, mult=ff_mult)

            self.layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, ff)
            ]))

        # weight tie projections across all layers

        # 跨层权重绑定：除首层外，各层注意力复用首层的查询/键打分投影
        first_block, _ = self.layers[0]
        for block, _ in self.layers[1:]:
            block.fn.to_q_attn_logits = first_block.fn.to_q_attn_logits
            block.fn.to_k_attn_logits = first_block.fn.to_k_attn_logits

        # to logits

        # 输出头：LayerNorm 后线性映射到词表大小 num_tokens
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(
        self,
        x,
        mask=None
    ):
        """x: (batch, 序列长度) 的 token id 整数张量；mask: (batch, 序列长度) bool 有效位掩码，None 时全视为有效。
        返回 (batch, 序列长度, num_tokens) 的预测 logits。"""
        n, device = x.shape[1], x.device
        if mask is None:
            mask = torch.ones_like(x).bool().to(device)
        # 查表得到 token 嵌入（不乘 sqrt(dim) 缩放因子）
        x = self.token_emb(x)
 
        # 绝对位置编码：按位置索引查表后加到嵌入上
        if exists(self.abs_pos_emb):
            pos_emb = self.abs_pos_emb(torch.arange(n, device=device))
            x = x + rearrange(pos_emb, 'n d -> () n d')

        # 逐块堆叠：注意力子层与前馈子层各自用残差相加（归一化已在块内前置）
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x

        return self.to_logits(x)