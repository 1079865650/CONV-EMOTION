import pickle
import torch
import os

# 1. 使用绝对路径，确保在哪运行都能找到
path = '/home/rock/project/conv-emotion/DialogueGCN-mianzhang/data/iemocap/ckpt/data.pkl'

if not os.path.exists(path):
    print(f"错误：找不到文件 {path}，请检查路径是否正确！")
else:
    with open(path, 'rb') as f:
        # 关键点：加上 encoding='latin1' 解决报错
        data = pickle.load(f, encoding='latin1')

    print("数据集包含的键:", data.keys())

    # 2. 查看样本维度
    # 注意：根据你之前 preprocess.py 的逻辑，data 是一个字典，包含 'train', 'dev', 'test'
    train_data = data['train']
    sample = train_data[0]

    print("\n--- 第一条样本信息 ---")
    # 打印 sample 对象的所有属性，看看哪个是特征
    print("样本属性:", dir(sample))
    
    # 尝试打印 text 特征的维度
    # 在 DialogueGCN 中，特征通常叫 text, visual, audio
    if hasattr(sample, 'text'):
        feat = torch.tensor(sample.text)
        print(f"文本特征 (text) 维度: {feat.shape}")
    
    if hasattr(sample, 'visual'):
        v_feat = torch.tensor(sample.visual)
        print(f"视觉特征 (visual) 维度: {v_feat.shape}")

    if hasattr(sample, 'audio'):
        a_feat = torch.tensor(sample.audio)
        print(f"音频特征 (audio) 维度: {a_feat.shape}")