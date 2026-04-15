import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, GraphConv

# ================= 新增：门控残差模块 (Gated Residual) =================
class GatedResidual(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GatedResidual, self).__init__()
        # 如果输入输出维度不一致（比如 200 -> 128），用一个线性层对齐
        # 如果一致（比如 128 -> 128），就直接用 Identity 恒等映射，不增加额外参数
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
        # 门控权重 W_g，输入是拼接后的维度 (in_dim + out_dim)，输出是 out_dim
        self.gate = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, h_in, h_gcn):
        # 1. 拼接原始特征和 GCN 提取的特征: [h^(l) || h^(l+1)_GCN]
        cat_h = torch.cat([h_in, h_gcn], dim=-1)
        
        # 2. 核心公式 1：计算门控变量 alpha (通过 Sigmoid 压缩到 0~1 之间)
        alpha = torch.sigmoid(self.gate(cat_h))
        
        # 3. 对齐原始特征的维度 (为了能跟 h_gcn 相加)
        h_proj = self.proj(h_in)
        
        # 4. 核心公式 2：加权融合 (保留多少个性，吸收多少共性，由模型自己说了算！)
        h_out = alpha * h_gcn + (1 - alpha) * h_proj
        
        return h_out
# =====================================================================

class GCN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, args):
        super(GCN, self).__init__()
        self.num_relations = 2 * args.n_speakers ** 2
        
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations, num_bases=30)
        self.conv2 = GraphConv(h1_dim, h2_dim)
        
        # 实例化两层门控残差网络，给每一层 GCN 都配上一个“智能阀门”
        self.gated_res1 = GatedResidual(g_dim, h1_dim)
        self.gated_res2 = GatedResidual(h1_dim, h2_dim)

    def forward(self, node_features, edge_index, edge_norm, edge_type):
        
        # ----------- 第一层 GCN 传播 -----------
        # 1. 备份输入特征
        h0 = node_features
        # 2. 如图卷积运算 (RGCN 不带 edge_norm，听你的，保持默认)
        h1_gcn = self.conv1(h0, edge_index, edge_type)
        # 3. 拦截并进行门控融合！
        h1 = self.gated_res1(h0, h1_gcn)
        # ---------------------------------------
        
        # ----------- 第二层 GCN 传播 -----------
        # 1. 这里 GraphConv 支持传入连续权重，把我们动态图的 edge_norm 喂进去！
        h2_gcn = self.conv2(h1, edge_index, edge_weight=edge_norm)
        # 2. 拦截并进行门控融合！
        h2 = self.gated_res2(h1, h2_gcn)
        # ---------------------------------------

        return h2