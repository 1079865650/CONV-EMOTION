import torch
import torch.nn as nn

class TextGuidedAttention(nn.Module):
    def __init__(self, dim=1024, num_heads=8):
        super(TextGuidedAttention, self).__init__()
        # 调用的多头注意力函数
        # batch_first=True 非常重要，它让输入维度保持 [batch_size, seq_len, dim]
        self.attn_audio = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.attn_video = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        

        # 把文本，音频，视觉三个1024维，但是显卡比较有压力，所以压缩回1024维   
        self.fusion_linear = nn.Linear(dim * 3, dim)
        self.relu = nn.ReLU()

    def forward(self, text, audio, video):
        """
        text, audio, video 的维度预期均为: [batch_size, seq_len, 1024]
        """
        # 1. 文本询问音频 (Text guides Audio)
        # 文本是 Query，音频是 Key 和 Value
        audio_aware, _ = self.attn_audio(query=text, key=audio, value=audio)
        
        # 2. 文本询问视觉 (Text guides Video)
        # 文本是 Query，视觉是 Key 和 Value
        video_aware, _ = self.attn_video(query=text, key=video, value=video)
        
        # 3. 特征拼接
        # 现在的 audio_aware 已经不是纯音频了，而是“与文本相关的音频重点”
        concat_feat = torch.cat([text, audio_aware, video_aware], dim=-1) # 维度变成 [batch, seq_len, 3072]
        
        # 4. 融合与降维
        final_feat = self.relu(self.fusion_linear(concat_feat)) # 维度变回 [batch, seq_len, 1024]
        
        return final_feat