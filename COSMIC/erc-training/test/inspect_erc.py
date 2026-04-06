import pickle
import numpy as np
import torch



print(torch.cuda.is_available())
print(torch.cuda.get_device_capability())


data_path = '/home/rock/project/conv-emotion/COSMIC/erc-training/data/erc-training/iemocap/iemocap_features_roberta.pkl'

with open(data_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# 锁定 Fold [2]（我们推测的特征桶）
features_fold = data[2]
first_id = list(features_fold.keys())[0]
feature_list = features_fold[first_id]

print(f"--- 最终底层探测 ---")
print(f"对话 ID: {first_id}")
print(f"该对话包含轮数: {len(feature_list)}")

# 取出第一轮话的特征
first_utt_feature = feature_list[0]

print(f"第一轮话的特征类型: {type(first_utt_feature)}")

if hasattr(first_utt_feature, 'shape'):
    print(f"🚀 特征维度 (Shape): {first_utt_feature.shape}")
    print(f"结论：你的模型输入层 $d_{{in}}$ 应设置为 {first_utt_feature.shape[-1]}")
elif isinstance(first_utt_feature, list):
    print(f"居然还是列表？长度为: {len(first_utt_feature)}")
    print(f"内容预览: {first_utt_feature[:5]}")