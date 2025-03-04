import torch
import torch.nn as nn
from data.pyg_dataToGraph import DataToGraph
from matplotlib import pyplot as plt
#%%
# TODO 加载数据集
dataset = DataToGraph(
    raw_data_path='../../data/',
    dataset_name='TFF' + '.mat')  # 格式: [(graph,label),...,(graph,label)]

input_dim = dataset[0].x.size(1)
num_classes = dataset.num_classes

# 提取所有的x和y
x0 = []
labels = []

for data in dataset:
    # 提取x (形状为 [num_nodes, input_dim])
    # 但是你提到dataset.x的形状是 [24,50]，这可能是一个图的x特征矩阵
    x0.append(data.x)
    # 提取y（标量标签）
    labels.append(data.y)

# 将列表转换为张量
x0 = torch.stack(x0)  # 形状 [num_samples, 24, 50]
labels = torch.stack(labels)  # 形状 [num_samples]

print(num_classes)
print("X0 shape:", x0.shape)
print("Labels shape:", labels.shape)
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
# 将数据传输到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO 确定超参数的值
# 超参数值
num_steps = 1000  # 假设扩散步数为 1000
eps = 1e-5  # 避免除以零或引入过小的数值的小偏移量

# 生成时间步的序列
t = torch.linspace(0, 1, num_steps + 1)  # 主要时间步范围从 0 到 1

# 使用余弦调度生成 betas
betas = torch.cos(torch.pi / 2.0 * t) ** 2  # 余弦平方函数
betas = betas / betas.max()  # 归一化到 0-1 范围
betas = torch.flip(betas, [0])  # 反转顺序，以确保从小到大递增
betas = torch.clamp(betas, min=1e-5, max=0.5e-2)  # 调整范围到 (1e-5, 0.5e-2)

# 计算 alpha , alpha_prod , alpha_prod_previous , alpha_bar_sqrt 等变量的值
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, dim=0)  # 累积连乘
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)  # p means previous
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

# 将超参数也移动到GPU
betas = betas.to(device)
alphas = alphas.to(device)
alphas_prod = alphas_prod.to(device)
alphas_prod_p = alphas_prod_p.to(device)
alphas_bar_sqrt = alphas_bar_sqrt.to(device)
one_minus_alphas_bar_log = one_minus_alphas_bar_log.to(device)
one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.to(device)

assert alphas_prod.shape == alphas_prod.shape == alphas_prod_p.shape \
       == alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape \
       == one_minus_alphas_bar_sqrt.shape
print("all the same shape:", betas.shape)
#%%
def q_sample(x0, t, noise):
    """前向扩散过程：根据时间步t给x0加噪"""
    sqrt_alpha_prod = torch.sqrt(alphas_prod[t]).view(-1, 1, 1)
    sqrt_one_minus_alpha_prod = torch.sqrt(1 - alphas_prod[t]).view(-1, 1, 1)
    return sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise

class MySequential(nn.Sequential):
    def forward(self, x, t_emb):
        for module in self:
            if isinstance(module, ConditionalBlock):  # 仅对特定模块传参
                x = module(x, t_emb)
            else:  # 其他模块按默认方式处理
                x = module(x)
        return x

import math
import torch


def sinusoidal_embedding(t, dim):
    """
    Args:
        t: 时间步张量 [batch_size, ]
        dim: 嵌入维度
    Returns:
        嵌入向量 [batch_size, dim]
    """
    device = t.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = t.float()[:, None] * emb[None, :]  # [batch_size, half_dim]

    # 拼接正弦和余弦分量
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # 处理奇数维度情况
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')

    return emb


class ConditionalEmbedding(nn.Module):
    def __init__(self, num_classes, time_dim=256, label_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        self.label_embed = nn.Embedding(num_classes, label_dim)
        self.fusion = nn.Sequential(
            nn.Linear(time_dim + label_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )

    def forward(self, t, y):
        # t: [B,] 时间步
        # y: [B,] 标签
        t_emb = sinusoidal_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)  # [B, time_dim]

        l_emb = self.label_embed(y).squeeze(1)  # [B, label_dim]

        # 融合时间与标签信息
        combined = torch.cat([t_emb, l_emb], dim=1)
        return self.fusion(combined)  # [B, time_dim]
#%%
class EnhancedTimeEmbedding(nn.Module):
    """增强时间嵌入（添加多层感知）"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.embed = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            #nn.SiLU(),
            nn.Linear(dim*4, dim),
            #nn.SiLU(),
            #nn.Linear(dim, dim)
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.embed(emb)
#%% md
### Conditional Block
#%%
from einops import rearrange

class ConditionalBlock(nn.Module):
    """基于你原有MyBlock改造的条件版本"""
    def __init__(self, in_ch, out_ch, cond_dim, mult=1):
        """
        Args:
            cond_dim: 条件向量的维度 (time+label的融合维度)
        """
        super().__init__()

        # 修改后的条件投影层（移除偏置项）验证条件注入的有效性
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, out_ch*2, bias=False),  # 关键修改：bias=False
            nn.GELU()
        )

        # 保持原有卷积结构
        self.ds_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.conv = nn.Sequential(
            nn.GroupNorm(1, out_ch),
            nn.Conv2d(out_ch, out_ch * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, out_ch * mult),
            nn.Conv2d(out_ch * mult, out_ch, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond_emb):
        """输入变化：t_emb → cond_emb (融合时间+标签的条件向量)"""
        h = self.ds_conv(x)

        # 条件注入 (scale and shift)
        scale, shift = self.cond_mlp(cond_emb).chunk(2, dim=1)  # [B, 2*out_ch] → [B, out_ch], [B, out_ch]
        h = h * (1 + scale[:, :, None, None])  # 缩放
        h = h + shift[:, :, None, None]        # 偏移

        h = self.conv(h)
        return h + self.res_conv(x)  # 保持原有残差连接

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

from torch import einsum, softmax
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b"
                     " h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

import torch.nn.functional as F
class UNetClassifier(nn.Module):
    def __init__(self, num_classes, time_dim=128):
        super().__init__()
        chs = [1, 64, 128, 256]  # 输入通道调整为1 (单通道特征图)

        # 时间嵌入层 (移除标签条件)
        self.time_embed = EnhancedTimeEmbedding(time_dim)

        # 下采样路径 (增强特征提取)
        self.down = nn.ModuleList([
            MySequential(
                ConditionalBlock(chs[i], chs[i+1], cond_dim=time_dim),
                ConditionalBlock(chs[i+1], chs[i+1], cond_dim=time_dim),
                Residual(PreNorm(chs[i+1], LinearAttention(chs[i+1])))
            ) for i in range(len(chs)-1)
        ])

        # 中间层 (适配分类任务)
        self.mid = MySequential(
            ConditionalBlock(chs[-1], chs[-1]*2, cond_dim=time_dim),
            Residual(PreNorm(chs[-1]*2, Attention(chs[-1]*2))),
            ConditionalBlock(chs[-1]*2, chs[-1], cond_dim=time_dim)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(chs[-1], 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, t):
        """
        输入:
            x: [B, H, W] (如 [B,24,50])
            t: [B,] 时间步
        """
        x = x.unsqueeze(1)  # [B,1,24,50]

        # 时间嵌入
        t_emb = self.time_embed(t)

        skips = []

        # 编码器
        for block in self.down:
            x = block(x, t_emb)
            skips.append(x)
            x = F.max_pool2d(x, kernel_size=(2,1))  # 保持宽度不变

        # 中间处理
        x = self.mid(x, t_emb)

        # 分类
        return self.classifier(x)