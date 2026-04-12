import pickle
import numpy as np

path = '/home/rock/project/conv-emotion/data/erc-training/iemocap/iemocap_features_roberta.pkl'
with open(path, 'rb') as f:
    raw_data = pickle.load(f)

# 【根据扫描结果精准赋值】
video_speakers = raw_data[0]   # Index 0 是说话人
video_labels   = raw_data[1]   # Index 1 是情绪标签
video_text     = raw_data[2]   # Index 2 是 1024维文本特征 (注意: 刚才扫描显示Index 2是1024维)
video_audio    = raw_data[3]   # Index 3
video_visual   = raw_data[4]   # Index 4
video_sentence = raw_data[6]   # Index 6 是原始文本
train_vids     = list(raw_data[7])
test_vids      = list(raw_data[9])

# 情感映射
emo_map = {0: 'Neutral', 1: 'Anger', 2: 'Happiness', 3: 'Sadness', 4: 'Surprise', 5: 'Fear'}

# 随便挑一个训练集视频展示
vid = train_vids[0]

print("="*80)
print(f"✅ 最终确认结构 - 视频 ID: {vid}")
print("="*80)

for i in range(min(3, len(video_sentence[vid]))):
    print(f"【句子 {i+1}】")
    print(f"  📝 文本: {video_sentence[vid][i]}")
    print(f"  👤 说话人: {video_speakers[vid][i]}")
    print(f"  🎭 标签: {video_labels[vid][i]} ({emo_map.get(video_labels[vid][i], 'Unknown')})")
    print(f"  🚀 特征维度: {np.array(video_text[vid][i]).shape}")
    print(f"  📈 特征数值(前3位): {video_text[vid][i][:3]}")
    print("-" * 40)

print(f"\n训练集数量: {len(train_vids)} | 测试集数量: {len(test_vids)}")