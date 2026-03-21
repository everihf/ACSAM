import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):#在判断：这个模块是不是 BatchNorm 层
            module.backup_momentum = module.momentum#备份原来的 momentum
            module.momentum = 0

    model.apply(_disable)
#把 BatchNorm 层的 momentum=0，临时冻结 running stats
#momentum 越大，running stats 更新得越快；momentum 越小，running stats 更新得越慢。momentum=0 就是完全冻结了 running stats，不更新了。
def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            #hasattr（）意思是：判断对象 obj 是否有一个叫 "attr_name" 的属性。返回值：有 → True；没有 → False
            module.momentum = module.backup_momentum
#把模型里 BatchNorm 层的 momentum 恢复成原来的值，让 BN 继续正常更新 running mean / running var。
#它的实现是：如果是 _BatchNorm 并且之前备份过 backup_momentum，就恢复回去
#running_mean←(1−m)⋅running_mean+m⋅μB​   running_var←(1−m)⋅running_var+m⋅σB^2​
    model.apply(_enable)


# #推理/测试时怎么办？
# 测试时通常一次只进来很少数据，甚至 batch size = 1。
# 这时候你不能再靠“当前 batch 的均值方差”了，因为不稳定。
# 所以 BatchNorm 会在训练过程中，顺便维护两组“长期统计量”：
# running_mean
# running_var
# 这两个就叫：
# running stats（运行统计量）
# 它们是对训练过程中各个 batch 的均值/方差做的滑动平均。

#训练时 model.train()
# BatchNorm 会做两件事：
# 1）归一化时使用当前 batch 的 mean/var
# 也就是用这一小批数据自己的统计量来标准化。
# 2）更新 running_mean / running_var
# 把当前 batch 的统计信息融入长期记忆。

#测试时 model.eval()
# BatchNorm 不再用当前 batch 的统计量，而是直接用训练期间积累的：
# running_mean
# running_var

#六、为什么 SAM 里经常要这么做？
# 这是你最该理解的部分。
# SAM 的一次更新不是普通的一次 forward/backward，而是两步：
# 第一次 forward/backward，算出梯度，找到“扰动方向”
# 把参数沿着这个方向扰动一下，再做第二次 forward/backward
# 问题在于：
# 这第二次 forward 不是在“真实参数”下的正常前向，而是在“被故意扰动后的参数”下做的。
# 如果这时候 BatchNorm 还照常更新 running stats，就会出问题。