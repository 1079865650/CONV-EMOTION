import numpy as np
import torch
import dgcn

log = dgcn.utils.get_logger()

def batch_graphify(features, lengths, speaker_tensor, wp, wf, edge_type_to_idx, att_model, device):
    # wp 和 wf 参数保留占位（防止外部调用报错），但内部已彻底废弃滑动窗口！
    
    node_features, edge_index, edge_norm, edge_type = [], [], [], []
    batch_size = features.size(0)
    length_sum = 0
    edge_index_lengths = []

    # ================= 🌟 炼丹核心参数 🌟 =================
    # tau 就是你论文里的阈值 \tau。
    # 建议你在外面的自动化脚本里测试这几个值：0.03, 0.05, 0.08
    tau = 0.05 
    # ======================================================

    # 1. 调用 EdgeAtt 获取动态语义打分矩阵 (注意参数变了)
    edge_weights_list = att_model(features, lengths)

    for j in range(batch_size):
        cur_len = lengths[j].item()
        node_features.append(features[j, :cur_len, :])

        # 拿出第 j 个对话的注意力矩阵 [cur_len, cur_len]
        A_ij = edge_weights_list[j]

        # 2. 数据驱动的动态连边！只保留 A_ij 大于阈值 tau 的边
        valid_edges = torch.nonzero(A_ij > tau) 
        perms = valid_edges.cpu().numpy().tolist()

        # 兜底机制：如果阈值设得太高导致图中没边了，强行加上自环(自己连自己)防报错
        if len(perms) == 0:
            perms = [(i, i) for i in range(cur_len)]

        perms_rec = [(item[0] + length_sum, item[1] + length_sum) for item in perms]
        length_sum += cur_len
        edge_index_lengths.append(len(perms))

        for item, item_rec in zip(perms, perms_rec):
            src, dst = item[0], item[1]
            edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))
            
            # 将算出来的高维语义相关度，直接作为图网络的边权重
            edge_norm.append(A_ij[src, dst])

            # 确定说话人关系类型
            speaker1 = speaker_tensor[j, src].item()
            speaker2 = speaker_tensor[j, dst].item()
            c = '0' if src < dst else '1'
            edge_type.append(edge_type_to_idx[str(speaker1) + str(speaker2) + c])

    node_features = torch.cat(node_features, dim=0).to(device)  # [E, D_g]
    edge_index = torch.stack(edge_index).t().contiguous().to(device)  # [2, E]
    edge_norm = torch.stack(edge_norm).to(device)  # [E]
    edge_type = torch.tensor(edge_type).long().to(device)  # [E]
    edge_index_lengths = torch.tensor(edge_index_lengths).long().to(device)  # [B]

    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths