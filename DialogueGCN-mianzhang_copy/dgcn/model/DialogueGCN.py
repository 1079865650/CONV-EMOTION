import torch
import torch.nn as nn

from .SeqContext import SeqContext
from .EdgeAtt import EdgeAtt
from .GCN import GCN
from .Classifier import Classifier
from .functions import batch_graphify
import dgcn

log = dgcn.utils.get_logger()

class DialogueGCN(nn.Module):
    def __init__(self, args):
        super(DialogueGCN, self).__init__()
        
        # ================= 核心修改点 =================
        # 这里的 u_dim 必须从 100 改为 1024，匹配你的 RoBERTa/Wav2Vec 特征维度！
        u_dim = 1024 
        # ==============================================
        
        g_dim = 200
        h1_dim = 100
        h2_dim = 100
        hc_dim = 100
        tag_size = 6

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        self.rnn = SeqContext(u_dim, g_dim, args)
        self.edge_att = EdgeAtt(g_dim, args)
        self.gcn = GCN(g_dim, h1_dim, h2_dim, args)
        self.clf = Classifier(g_dim + h2_dim, hc_dim, tag_size, args)

        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)

    def get_rep(self, data):
        # ================= 核心修改点 =================
        # 原版这里只传了 text_tensor。现在我们把 audio 和 video 也传进去！
        node_features = self.rnn(
            data["text_len_tensor"], 
            data["text_tensor"],
            data["audio_tensor"],   # 新增传入音频
            data["video_tensor"]    # 新增传入视觉
        ) 
        # ==============================================

        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
            node_features, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx, self.edge_att, self.device)

        graph_out = self.gcn(features, edge_index, edge_norm, edge_type)

        return graph_out, features

    def forward(self, data):
        graph_out, features = self.get_rep(data)
        out = self.clf(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])
        return out

    def get_loss(self, data):
        graph_out, features = self.get_rep(data)
        loss = self.clf.get_loss(torch.cat([features, graph_out], dim=-1),
                                 data["label_tensor"], data["text_len_tensor"])
        return loss