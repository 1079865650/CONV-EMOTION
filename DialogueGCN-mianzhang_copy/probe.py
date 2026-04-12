import pickle
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import dgcn

print("正在加载数据，准备照妖镜测试...")
data = pickle.load(open('data/iemocap/ckpt/data.pkl', 'rb'))

# 1. 扁平化提取：把所有对话里的所有【句子】全拆散，排成一列长队！
train_texts = []
train_labels = []
for s in data['train']:
    train_texts.append(torch.tensor(s.text))   # 比如 [47, 1024]
    train_labels.append(torch.tensor(s.label)) # 比如 [47] 对应的标签

test_texts = []
test_labels = []
for s in data['test']:
    test_texts.append(torch.tensor(s.text))
    test_labels.append(torch.tensor(s.label))

# 使用 torch.cat 首尾相接
X_train = torch.cat(train_texts, dim=0)
y_train = torch.cat(train_labels, dim=0).long()

X_test = torch.cat(test_texts, dim=0)
y_test = torch.cat(test_labels, dim=0).long()

# ================= 🚑 维度抢救中心 =================
# 如果标签变成了二维的 [4717, 1]，强行把它拍扁成一维的 [4717]
if len(y_train.shape) > 1:
    if y_train.shape[1] == 1:
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
    else:
        # 如果是 One-Hot 编码 [4717, X]，就提取正确答案的索引
        y_train = torch.argmax(y_train, dim=1)
        y_test = torch.argmax(y_test, dim=1)
# ====================================================

# ================= 🏷️ 标签终极清洗中心 =================
# 1. 找出训练集里到底有多少种奇葩标签编号
unique_labels = torch.unique(y_train)
num_classes = len(unique_labels)
print(f"检测到 {num_classes} 种真实标签编号: {unique_labels.tolist()}")

# 2. 建立一个映射字典，把比如 61, 62 这种数字，强行映射回标准的 0, 1, 2...
label_mapping = {val.item(): idx for idx, val in enumerate(unique_labels)}

# 3. 重新给 train 和 test 换上标准标签
y_train = torch.tensor([label_mapping[val.item()] for val in y_train]).long()
# 如果测试集蹦出没见过的怪标签，默认给0防止报错
y_test = torch.tensor([label_mapping.get(val.item(), 0) for val in y_test]).long() 
# ========================================================

print(f"标签清洗完毕！X_train: {X_train.shape}, y_train: {y_train.shape}")

# 2. 定义一个史上最弱的模型：只有一层全连接层
class DumbClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # ！！！这里改成动态适应类别数量！！！
        self.fc = nn.Linear(1024, num_classes) 
        
    def forward(self, x):
        return self.fc(x)

# ======== 下面的 model = DumbClassifier() 和训练代码不用动 ========