import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_crossentropy(pred, label, smoothing=0.1):#pred 是模型的输出，label 是标签，smoothing 是标签平滑的参数 ε=0.1
    n_class = pred.size(1)#类别数

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))#每个类 = ε / (K-1)，
    one_hot.scatter_(dim=1, index=label.unsqueeze(1), value=1.0 - smoothing)#把正确类改成：1 - ε
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)#KL散度
# KL(target||input)=CE(target,input)−H(target)  KL散度=交叉熵损失-真实标签的熵（常数），所以等价于交叉熵损失
