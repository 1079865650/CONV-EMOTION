import pickle
import dgcn # 必须导入，否则读不了 dgcn.Sample 对象

# 读取你刚才训练用的那个 pkl 文件
data = pickle.load(open('data/iemocap/ckpt/data.pkl', 'rb'))

# 提取测试集中所有对话的 ID
test_vids = [sample.vid for sample in data['test']]

# 提取出所有的 Session 前缀 (比如 'Ses01', 'Ses05') 并去重
sessions = set([vid[:5] for vid in test_vids])

print("============== 审查结果 ==============")
print(f"你的测试集里包含了以下 Session 的数据: {sorted(list(sessions))}")
print("======================================")