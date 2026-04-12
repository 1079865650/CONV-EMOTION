#!/bin/bash
# 自动化炼丹脚本 (带自动建档与日志归档功能)

# 1. 定义日志存放的文件夹名
LOG_DIR="train_log"

# 2. 检查文件夹是否存在，如果不存在就自动新建它 (-p 参数保证了不会报错)
mkdir -p $LOG_DIR

# 3. 把生成的日志文件放进刚才定义的文件夹里
LOG_FILE="$LOG_DIR/search_results_$(date +%m%d_%H%M).log"

echo "🚀 开始自动化调参... 所有输出将同步保存在: $LOG_FILE" | tee -a $LOG_FILE

# 测试三种不同的学习率和两种隐藏层维度
for lr in 0.0002 0.0001 0.00005
do
    for hs in 128 200 300
    do
        echo -e "\n=========================================" | tee -a $LOG_FILE
        echo "🔥 正在测试组合 -> LR: $lr, Hidden Size: $hs" | tee -a $LOG_FILE
        echo "=========================================" | tee -a $LOG_FILE
        
        # 执行训练脚本，并将输出追加到统一的日志文件中
        python train.py --data data/iemocap/ckpt/iemocap_roberta_1024_v1.pkl \
                        --device cuda \
                        --epochs 60 \
                        --batch_size 32 \
                        --learning_rate $lr \
                        --hidden_size $hs \
                        --drop_rate 0.4 \
                        --rnn lstm \
                        --from_begin 2>&1 | tee -a $LOG_FILE
    done
done

echo "✅ 所有组合测试完毕！请前往 $LOG_DIR 文件夹查看完整日志。" | tee -a $LOG_FILE


#   正在测试组合 -> LR: 0.0002, Hidden Size: 128
# dev: 100%|██████████| 1/1 [00:00<00:00,  1.15it/s]
# dev: 100%|██████████| 1/1 [00:00<00:00,  1.15it/s]
# 04/12/2026 10:41:52 [Dev set] [f1 0.6451]

# test:   0%|          | 0/1 [00:00<?, ?it/s]/home/rock/project/conv-emotion/DialogueGCN-mianzhang_copy/dgcn/model/EdgeAtt.py:37: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
#   probs = F.softmax(score)  # [L']

# test: 100%|██████████| 1/1 [00:01<00:00,  1.30s/it]
# test: 100%|██████████| 1/1 [00:01<00:00,  1.30s/it]
# 04/12/2026 10:41:53 [Test set] [f1 0.5683]