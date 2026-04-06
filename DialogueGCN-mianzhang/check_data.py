import pickle
import torch

path = '/home/rock/project/conv-emotion/data/erc-training/iemocap/iemocap_features_roberta.pkl' 

with open(path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# 提取 video_text (索引为 3)
video_text = data[3]
video_audio = data[4]
video_visual = data[5]

# 随便找一个视频 ID
first_vid = list(video_text.keys())[0]

print(f"--- 视频 ID: {first_vid} ---")
print(f"🚀 文本特征 (RoBERTa) 维度: {torch.tensor(video_text[first_vid]).shape}")
print(f"🚀 音频特征 维度: {torch.tensor(video_audio[first_vid]).shape}")
print(f"🚀 视觉特征 维度: {torch.tensor(video_visual[first_vid]).shape}")