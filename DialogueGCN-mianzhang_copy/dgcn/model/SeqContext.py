import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ================= 新增：跨模态注意力模块 =================
class TextGuidedAttention(nn.Module):
    def __init__(self, dim=1024, num_heads=8):
        super(TextGuidedAttention, self).__init__()
        # PyTorch 自带的多头注意力机制
        self.attn_audio = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.attn_video = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        # 降维层：把拼接后的 3072 维压回 1024 维，方便无缝接入后面的 RNN
        self.fusion_linear = nn.Linear(dim * 3, dim)
        self.relu = nn.ReLU()

    def forward(self, text, audio, video):
        # 文本作为 Query 引导音频和视觉
        audio_aware, _ = self.attn_audio(query=text, key=audio, value=audio)
        video_aware, _ = self.attn_video(query=text, key=video, value=video)
        
        # 拼接特征 [batch, seq_len, 3072]
        concat_feat = torch.cat([text, audio_aware, video_aware], dim=-1)
        
        # 融合降维 [batch, seq_len, 1024]
        final_feat = self.relu(self.fusion_linear(concat_feat))
        return final_feat
# ==========================================================

class SeqContext(nn.Module):
    def __init__(self, u_dim, g_dim, args):
        super(SeqContext, self).__init__()
        self.input_size = u_dim
        self.hidden_dim = g_dim
        
        # 实例化我们刚才写的注意力模块 (假设输入特征都是 1024 维)
        self.cross_modal_attn = TextGuidedAttention(dim=self.input_size)

        if args.rnn == "lstm":
            self.rnn = nn.LSTM(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                               bidirectional=True, num_layers=2, batch_first=True)
        elif args.rnn == "gru":
            self.rnn = nn.GRU(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                              bidirectional=True, num_layers=2, batch_first=True)

    # ！！！注意这里：函数参数增加了 audio_tensor 和 video_tensor ！！！
    def forward(self, text_len_tensor, text_tensor, audio_tensor, video_tensor):
        
        # 1. 在进入序列模型前，先进行跨模态融合！
        fused_tensor = self.cross_modal_attn(text_tensor, audio_tensor, video_tensor)

        # 2. 将融合后高维特征打包送入 RNN（这里用 fused_tensor 替换了原来的 text_tensor）
        packed = pack_padded_sequence(
            fused_tensor,
            text_len_tensor.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        rnn_out, (_, _) = self.rnn(packed, None)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        return rnn_out