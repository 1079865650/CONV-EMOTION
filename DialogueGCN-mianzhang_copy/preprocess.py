import argparse
from tqdm import tqdm
import pickle
import dgcn

log = dgcn.utils.get_logger()

def split():
    dgcn.utils.set_seed(args.seed)

    # 1. 加载原始数据
    # path = '/home/rock/project/conv-emotion/data/erc-training/iemocap/iemocap_features_roberta.pkl'
    path = '/home/rock/project/conv-emotion/data/erc-training/iemocap/iemocap_features_whisper_roberta.pkl'
    raw_data = pickle.load(open(path, 'rb'), encoding='latin1')
    print(f"😎 成功读取！新数据集里共有 {len(raw_data)} 个大项元素！")

    # 2. 【核心重写：精准索引赋值】
    # 根据之前的测试结果，映射关系必须严格如下：
    video_speakers = raw_data[0]   # Index 0: 说话人 (M/F)
    video_labels   = raw_data[1]   # Index 1: 情绪标签 (0-5 整数)
    video_text     = raw_data[2]   # Index 2: 1024维文本特征 (RoBERTa)
    video_audio    = raw_data[3]   # Index 3: 音频特征
    video_visual   = raw_data[4]   # Index 4: 视觉特征
    video_sentence = raw_data[6]   # Index 6: 原始文本内容
    trainVids      = raw_data[7]   # Index 7: 训练集 ID 列表
    test_vids      = raw_data[9]   # Index 9: 测试集 ID 列表 (注意是 9)

    # 3. 划分开发集 (从训练集中切出 10%)
    train, dev, test = [], [], []
    trainVids = sorted(list(trainVids))
    dev_size = int(len(trainVids) * 0.1)
    train_vids_final, dev_vids_final = trainVids[dev_size:], trainVids[:dev_size]

    # 4. 封装成 dgcn 识别的 Sample 对象
    # 训练集
    for vid in tqdm(train_vids_final, desc="Processing train"):
        train.append(dgcn.Sample(vid, video_speakers[vid], video_labels[vid],
                                 video_text[vid], video_audio[vid], video_visual[vid],
                                 video_sentence[vid]))
    # 开发集
    for vid in tqdm(dev_vids_final, desc="Processing dev"):
        dev.append(dgcn.Sample(vid, video_speakers[vid], video_labels[vid],
                               video_text[vid], video_audio[vid], video_visual[vid],
                               video_sentence[vid]))
    # 测试集
    for vid in tqdm(test_vids, desc="Processing test"):
        test.append(dgcn.Sample(vid, video_speakers[vid], video_labels[vid],
                                video_text[vid], video_audio[vid], video_visual[vid],
                                video_sentence[vid]))

    log.info(f"Train/Dev/Test split done: {len(train)}/{len(dev)}/{len(test)}")
    return train, dev, test

def main(args):
    train, dev, test = split()
    data = {"train": train, "dev": dev, "test": test}
    dgcn.utils.save_pkl(data, args.data)
    print(f"🎉 预处理完成！数据已保存至: {args.data}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")
    parser.add_argument("--data", type=str, required=True, help="Output path for data.pkl")
    parser.add_argument("--dataset", type=str, required=True, choices=["iemocap", "avec", "meld"])
    parser.add_argument("--seed", type=int, default=24, help="Random seed")
    args = parser.parse_args()
    main(args)