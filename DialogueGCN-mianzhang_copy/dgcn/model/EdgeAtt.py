import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeAtt(nn.Module):
    def __init__(self, g_dim, args):
        super(EdgeAtt, self).__init__()
        self.device = args.device
        
        # ================= 动态构图核心参数 =================
        # 对应公式里的 W_a，输入是两个节点特征的拼接 (g_dim * 2)，输出是一个标量打分
        self.W_a = nn.Linear(g_dim * 2, 1)
        # 对应公式里的 LeakyReLU
        self.leaky_relu = nn.LeakyReLU(0.2)
        # ====================================================

    def forward(self, node_features, text_len_tensor):
        # 注意：我们去掉了原版没用的 edge_ind 参数
        B, mx_len, D_g = node_features.size()
        alphas = []

        # 1. 矩阵膨胀操作：计算图中任意两个节点 i 和 j 的关系
        # h_i 变成 [B, L, L, D_g]
        h_i = node_features.unsqueeze(2).expand(B, mx_len, mx_len, D_g)
        # h_j 变成 [B, L, L, D_g]
        h_j = node_features.unsqueeze(1).expand(B, mx_len, mx_len, D_g)
        
        # 2. 将它们拼接在一起: [h_i || h_j]，维度变成 [B, L, L, D_g * 2]
        cat_h = torch.cat([h_i, h_j], dim=-1)

        # 3. 核心公式 1: e_{ij} = LeakyReLU(W_a [h_i || h_j])
        e_ij = self.leaky_relu(self.W_a(cat_h).squeeze(-1)) # 维度: [B, L, L]

        # 4. 遍历每个 Batch，截取实际对话长度并算 Softmax
        for i in range(B):
            cur_len = text_len_tensor[i].item()
            # 截取出真实存在的对话节点矩阵 [cur_len, cur_len]
            valid_e_ij = e_ij[i, :cur_len, :cur_len]
            
            # 核心公式 2: A_{ij} = Softmax(e_{ij})
            # 让模型自己算出每句话和其余所有话的相关度（和为1）
            A_ij = F.softmax(valid_e_ij, dim=-1) 
            
            alphas.append(A_ij)

        return alphas