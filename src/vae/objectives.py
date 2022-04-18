import torch
import numpy as np
from vae.utils import log_mean_exp, kl_divergence, check_action_ok, cls_to_label


def loss_fn(output, target, ltype):
    if ltype == "bce":
        output = output.loc
        assert torch.min(target.reshape(-1)) >= 0 and torch.max(target.reshape(-1)) <= 1, "Cannot use bce on data which is not normalised"
        loss = -torch.nn.functional.binary_cross_entropy(output.squeeze().cpu(), target.float().cpu().detach(), reduction="sum").cuda()
    elif ltype == "lprob":
        loss = output.log_prob(target.cuda()).view(*target.shape[:2], -1).sum(-1).sum(-1).sum(-1).double()
    elif ltype == "l1":
        l = torch.nn.L1Loss(reduction="sum")
        loss = -l(output.loc.cuda(), target.float().cuda().detach())
    elif ltype == "mse":
        l = torch.nn.MSELoss(reduction="sum")
        loss = -l(output.loc.cuda(), target.float().cuda().detach())
    return loss

def normalize(target, data=None):
    t_size= target.size()
    maxv, minv = torch.max(target.reshape(-1)), torch.min(target.view(-1))
    output = [torch.div(torch.add(target.reshape(-1), torch.abs(minv)), (maxv-minv)).reshape(t_size)]
    if data is not None:
        d_size = data.size()
        data_norm = torch.clamp(torch.div(torch.add(data.reshape(-1), torch.abs(minv)), (maxv-minv)), min=0, max=1)
        output.append(data_norm.reshape(d_size))
    return output


def elbo(model, x, K=1, ltype="lprob", eval=False):
    """Computes E_{p(x)}[ELBO] """
    qz_x, px_z, _ = model(x, K)
    d = x[0] if isinstance(x, list) else x
    lpx_z = loss_fn(px_z, d.repeat(K, *([1] *(len(d.shape)-1))), ltype=ltype)/K
    kld = kl_divergence(qz_x, model.pz(*model.pz_params))
    label = cls_to_label(model, int(x[1][0]))
    #scale = np.clip(10 ** (len(str(int(-lpx_z.sum(-1))))-3), 1, None)
    success = []
    if eval:
        for i, dif in enumerate(["normal", "easy", "hard"]):
            metrics = check_action_ok(np.asarray(px_z.loc.detach().cpu()).squeeze(-2), [label] * len(d) * K,  difficulty=dif)
            success.append(sum(metrics) / len(metrics))
    return -(lpx_z.sum(-1) - kld.sum().sum()), kld.sum(), [-lpx_z.sum()], success
