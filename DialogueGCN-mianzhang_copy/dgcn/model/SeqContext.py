import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ================= 新增：跨模态注意力模块 =================
class TextGuidedAttention(nn.Module):
    def __init__(self, dim=1024, num_heads=8):
        super(TextGuidedAttention, self).__init__()
        self.attn_audio = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.attn_video = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        # 【修改点1】只对音频和视频的融合结果进行降维，保护原始文本
        self.fusion_linear = nn.Linear(dim * 2, dim) 
        
        # 【修改点2】引入 LayerNorm，替换掉会抹杀负数的 ReLU
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, text, audio, video):
        # 1. 文本作为 Query 引导音频和视觉
        audio_aware, _ = self.attn_audio(query=text, key=audio, value=audio)
        video_aware, _ = self.attn_video(query=text, key=video, value=video)
        
        # 2. 【核心优化】只把提取到的音视频有效信息拼接并降维
        # 维度变化: [batch, seq_len, 2048] -> [batch, seq_len, 1024]
        av_feat = self.fusion_linear(torch.cat([audio_aware, video_aware], dim=-1))
        
        # 3. 【绝对防御：残差连接】(非常关键！)
        # 把原始的、纯净的 RoBERTa 文本特征，直接加上音视频的辅助特征
        # 这样即使音视频全是垃圾，模型大不了把 av_feat 变成 0，保底还有纯文本特征！
        final_feat = self.layer_norm(text + av_feat)
        
        return final_feat

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
        # ❌ 1. 把你加的高大上的跨模态注意力融合给“注释掉”（封印你的绝招）
        fused_tensor = self.cross_modal_attn(text_tensor, audio_tensor, video_tensor)

        # ✅ 2. 换成最粗暴的直接拼接 (text, audio, video 拼在一起，维度变成 3072)
        # ✅ 3. 借用注意力模块里的降维层，强行把 3072 维压回 1024 维，防止后面的 RNN 报错
        # raw_concat = torch.cat([text_tensor, audio_tensor, video_tensor], dim=-1)
        # fused_tensor = self.cross_modal_attn.relu(self.cross_modal_attn.fusion_linear(raw_concat))

        # 只用文本特征，不适用特征融合
        # fused_tensor  = text_tensor

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