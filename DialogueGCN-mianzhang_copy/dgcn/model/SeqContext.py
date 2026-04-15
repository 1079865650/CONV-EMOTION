import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ================= 新增：跨模态注意力模块 =================
class TextGuidedAttention(nn.Module):
    def __init__(self, dim=1024, num_heads=4): # 头数改为 4
        super(TextGuidedAttention, self).__init__()
        self.attn_audio = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.attn_video = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        self.fusion_linear = nn.Linear(dim * 2, dim) 
        self.dropout = nn.Dropout(0.3) # 增加 Dropout
        self.layer_norm = nn.LayerNorm(dim)
        
        # 【新增：自适应门控】初期为 0，让模型保底用文本，后期自动学习增加音视频权重
        self.alpha = nn.Parameter(torch.zeros(1)) 

    def forward(self, text, audio, video):
        # 1. 引导式注意力
        audio_aware, _ = self.attn_audio(query=text, key=audio, value=audio)
        video_aware, _ = self.attn_video(query=text, key=video, value=video)
        
        # 2. 融合与降维
        av_feat = self.fusion_linear(torch.cat([audio_aware, video_aware], dim=-1))
        av_feat = self.dropout(av_feat) # 增加随机性，防止死记硬背特征
        
        # 3. 【带权重的残差连接】
        # tanh(alpha) 保证了初始权重在 0 附近，不会在一开始就污染文本特征
        final_feat = self.layer_norm(text + torch.tanh(self.alpha) * av_feat)
        
        return final_feat

class SeqContext(nn.Module):
    def __init__(self, u_dim, g_dim, args):
        super(SeqContext, self).__init__()
        self.input_size = u_dim
        self.hidden_dim = g_dim

        # 【最推荐配置】：维度投影层
        # 将 Whisper 的 1280 映射为 1024
        self.audio_proj = nn.Linear(1280, self.input_size)
        # 将视觉的 512 映射为 1024
        self.video_proj = nn.Linear(1024, self.input_size)
        # 注意力模块，dim 也是 1024
        self.cross_modal_attn = TextGuidedAttention(dim=self.input_size)

        if args.rnn == "lstm":
            self.rnn = nn.LSTM(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                               bidirectional=True, num_layers=2, batch_first=True)
        elif args.rnn == "gru":
            self.rnn = nn.GRU(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                              bidirectional=True, num_layers=2, batch_first=True)

    # ！！！注意这里：函数参数增加了 audio_tensor 和 video_tensor ！！！
    def forward(self, text_len_tensor, text_tensor, audio_tensor, video_tensor):

        # 1. 维度对齐：把音频和视频都统一到 1024 维
        a_mapped = self.audio_proj(audio_tensor) # [batch, seq, 1024]
        v_mapped = self.video_proj(video_tensor) # [batch, seq, 1024]

        # 2. 跨模态注意力融合 (现在三个输入都是 1024，可以相加和计算了)
        fused_tensor = self.cross_modal_attn(text_tensor, a_mapped, v_mapped)
        
        # 1. 在进入序列模型前，先进行跨模态融合！
        # ❌ 1. 把你加的高大上的跨模态注意力融合给“注释掉”（封印你的绝招）
        # fused_tensor = self.cross_modal_attn(text_tensor, audio_tensor, video_tensor)

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
        rnn_out, _ = self.rnn(packed, None)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        return rnn_out