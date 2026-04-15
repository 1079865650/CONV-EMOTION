import torch
import torch.nn as nn
import torch.nn.functional as F

import dgcn

log = dgcn.utils.get_logger()

# ================= 新增：专门适配你代码的 FocalLoss =================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # 核心：复用你原来的 NLLLoss，但是设定 reduction='none' 以便逐个样本计算
        self.nll_loss = nn.NLLLoss(weight=weight, reduction='none')

    def forward(self, log_prob, target):
        # 1. 计算每个样本的基础 Loss (如果传了weight，这里就已经带上类别权重了)
        ce_loss = self.nll_loss(log_prob, target)
        
        # 2. 从 log_prob 中巧妙地恢复出模型预测对的真实概率 p_t
        # 因为 nll_loss 算出来其实就是 -log(p_t)，所以 e^(-ce_loss) 就是 p_t (假设没有weight时)
        # 为了更稳妥，我们直接取正确的维度概率：
        pt = torch.exp(log_prob.gather(1, target.unsqueeze(1)).squeeze(1))
        
        # 3. 焦点惩罚：预测得越准(pt接近1)，惩罚越狠(接近0)，逼迫算力留给困难样本
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()
# ====================================================================

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_size, tag_size, args):
        super(Classifier, self).__init__()
        self.emotion_att = MaskedEmotionAtt(input_dim)
        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.drop = nn.Dropout(args.drop_rate)
        self.lin2 = nn.Linear(hidden_size, tag_size)
        
        # ----------- 替换了原来的 NLLLoss 初始化 -----------
        loss_weights = None
        if args.class_weight:
            loss_weights = torch.tensor([1 / 0.086747, 1 / 0.144406, 1 / 0.227883,
                                         1 / 0.160585, 1 / 0.127711, 1 / 0.252668]).to(args.device)
            
        # 实例化我们的杀手锏：Focal Loss (同时继承了你优秀的 class_weight)
        self.loss_fn = FocalLoss(gamma=2.0, weight=loss_weights)
        # ---------------------------------------------------

    def get_prob(self, h, text_len_tensor):
        # h_hat = self.emotion_att(h, text_len_tensor)
        # hidden = self.drop(F.relu(self.lin1(h_hat)))
        hidden = self.drop(F.relu(self.lin1(h)))
        scores = self.lin2(hidden)
        log_prob = F.log_softmax(scores, dim=-1)

        return log_prob

    def forward(self, h, text_len_tensor):
        log_prob = self.get_prob(h, text_len_tensor)
        y_hat = torch.argmax(log_prob, dim=-1)

        return y_hat

    def get_loss(self, h, label_tensor, text_len_tensor):
        log_prob = self.get_prob(h, text_len_tensor)
        
        # ----------- 把 nll_loss 替换成了 loss_fn -----------
        loss = self.loss_fn(log_prob, label_tensor)
        # ---------------------------------------------------

        return loss

class MaskedEmotionAtt(nn.Module):
    def __init__(self, input_dim):
        super(MaskedEmotionAtt, self).__init__()
        self.lin = nn.Linear(input_dim, input_dim)

    def forward(self, h, text_len_tensor):
        batch_size = text_len_tensor.size(0)
        x = self.lin(h)  # [node_num, H]
        ret = torch.zeros_like(h)
        s = 0
        for bi in range(batch_size):
            cur_len = text_len_tensor[bi].item()
            y = x[s: s + cur_len]
            z = h[s: s + cur_len]
            scores = torch.mm(z, y.t())  # [L, L]
            probs = F.softmax(scores, dim=1)
            out = z.unsqueeze(0) * probs.unsqueeze(-1)  # [1, L, H] x [L, L, 1] --> [L, L, H]
            out = torch.sum(out, dim=1)  # [L, H]
            ret[s: s + cur_len, :] = out
            s += cur_len

        return ret