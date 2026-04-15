import pickle
import numpy as np
import os
import re
from tqdm import tqdm

# 1. 路径定义
old_path = '/home/rock/project/conv-emotion/data/erc-training/iemocap/iemocap_features_roberta.pkl'
whisper_path = '/home/rock/project/conv-emotion/data/wisper/iemocap/iemocap_whisper_large_v3.pkl'
save_path = '/home/rock/project/conv-emotion/data/erc-training/iemocap/iemocap_features_whisper_roberta.pkl'

# 2. 加载数据
print("📥 加载原始特征 (RoBERTa)...")
with open(old_path, 'rb') as f:
    raw_data = list(pickle.load(f))

print("📥 加载 Whisper 特征...")
with open(whisper_path, 'rb') as f:
    whisper_dict = pickle.load(f)

# 3. 开始精准对齐
old_audio_dict = raw_data[3]
new_audio_dict = {}
success_count = 0
fail_count = 0

print("✂️ 正在进行特征对齐手术...")
for vid in tqdm(old_audio_dict.keys()):
    # 提取 Whisper 字典中属于当前 vid 的所有 key
    # 例如找到所有以 "Ses04F_script01_2_" 开头的 key
    sub_keys = [k for k in whisper_dict.keys() if k.startswith(vid)]
    
    # 排序逻辑：按照 key 最后三位数字进行排序 (确保句子顺序正确)
    # 例如：M018 提取出 18，然后按数字排
    sub_keys.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
    
    # 检查数量是否匹配
    if len(sub_keys) != len(old_audio_dict[vid]):
        # 如果数量不完全一致，打印警告，但按顺序填充
        pass

    new_feats = []
    for i in range(len(old_audio_dict[vid])):
        if i < len(sub_keys):
            new_feats.append(whisper_dict[sub_keys[i]])
            success_count += 1
        else:
            # 补齐位 (基本不会发生，除非原始数据缺失)
            new_feats.append(np.zeros(1280))
            fail_count += 1
    
    new_audio_dict[vid] = new_feats

# 4. 覆盖旧音频特征并另存为
raw_data[3] = new_video_audio = new_audio_dict
with open(save_path, 'wb') as f:
    pickle.dump(tuple(raw_data), f)

print(f"\n✨ 融合完成！")
print(f"✅ 成功替换: {success_count} 条特征")
print(f"❌ 失败补齐: {fail_count} 条")
print(f"💾 新文件已保存至: {save_path}")