import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()#因为这里的 p.add_(e_w) 只是优化器手动改参数，不是神经网络前向传播的一部分。
    def first_step(self, zero_grad=False):#等价于 def first_step(,,,):with torch.no_grad():
        grad_norm = self._grad_norm() #||∇​L(w)||_2​  2阶梯度范数
        for group in self.param_groups:
            #PyTorch 优化器里经常有多个 param_group，每组可以有不同超参数，
            #比如：不同学习率，不同 weight decay，不同 rho，是否 adaptive，这里就是逐组处理。
            scale = group["rho"] / (grad_norm + 1e-12)#缩放系数 ρ / ||∇​L(w)||

            for p in group["params"]:#遍历这一组里的每个参数
                if p.grad is None: continue#如果某个参数没有梯度，就跳过
                self.state[p]["old_p"] = p.data.clone()#备份原来的p.如果不用clone:后面 p 改了，old_p 也会跟着变。
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)#表示把 scale 放到和参数 p 同样的设备/类型上。
                #adaptive为False时，e_w = p.grad * scale ; adaptive为True（自适应）时，e_w = torch.pow(p, 2) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e_w" "p+e_p"

        if zero_grad: self.zero_grad()
        #如果 zero_grad=True，就把当前梯度清空。
#为什么通常要清空？因为 first_step() 用的是第一次 backward 得到的梯度。
# 接下来在 𝑤+𝑒_𝑤处还要再 forward + backward 一次，得到第二次梯度。为了避免两次梯度累加，通常这里会清零。

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def _grad_norm(self):#把所有参数的梯度展平拼成一个超长向量，再求整体二范数
        shared_device = self.param_groups[0]["params"][0].device  # 将所有内容放在同一设备上，以防模型并行.拿第一个参数所在的设备，作为统一设备。
        norm = torch.norm(
                    torch.stack([#stack 的 tensor 必须在同一设备上。如果模型并行或不同参数在不同设备，不先统一会报错。
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)#||grad_p||_2,梯度的2范数
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    #把每个参数张量的局部范数组合成一个向量 [||gp1​||_2​,||gp2​​||_2​,…]
                    p=2  #再求整体2范数
               )
        return norm #||∇​L(w)||2​
    
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()


    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
