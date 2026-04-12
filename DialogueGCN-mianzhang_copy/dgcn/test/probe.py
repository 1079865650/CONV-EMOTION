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

# 使用 torch.cat 首尾相接，变成 [全部句子总数, 1024]
X_train = torch.cat(train_texts, dim=0)
y_train = torch.cat(train_labels, dim=0).long()

X_test = torch.cat(test_texts, dim=0)
y_test = torch.cat(test_labels, dim=0).long()

print(f"提取完毕！训练集共有 {X_train.shape[0]} 句话，测试集共有 {X_test.shape[0]} 句话。")

# 2. 定义一个史上最弱的模型：只有一层全连接层（没有任何上下文，没有图网络）
class DumbClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 6) # IEMOCAP 一般是 6 分类
        
    def forward(self, x):
        return self.fc(x)

model = DumbClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 3. 极速训练 30 个 Epoch
print("开始极速训练 (只有一层网络)...")
for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# 4. 在测试集上开奖！
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    predictions = torch.argmax(test_outputs, dim=1)
    # 计算加权 F1 分数
    f1 = f1_score(y_test.numpy(), predictions.numpy(), average='weighted')

print("================= 照妖镜审判结果 =================")
print(f"最弱智的单层模型，考出的 F1 分数是: {f1:.4f}")
print("==================================================")
if f1 > 0.85:
    print("🚨 实锤警报：特征严重泄露！这1024维数据里已经写满答案了！")
else:
    print("✅ 恭喜：特征很干净！你之前的模型确实是个天才！")