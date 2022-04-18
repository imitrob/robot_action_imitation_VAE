import math
import os, csv
import shutil
import time
import torch.distributions as dist
import numpy as np
import glob, imageio
import statistics
import math

def get_percentage_armpose(pose, pose_idx, above_threshold=50):
    if len(pose[0]) < pose_idx+1:
        return 1
    data = [int(s[pose_idx]) for s in pose]
    return sum([x > above_threshold for x in data]) / len(data)

def stdev(data, idx):
    data = [float(s[idx]) for s in data]
    return statistics.stdev(data)

def mean(data, idx):
    data = [float(s[idx]) for s in data]
    return statistics.mean(data)

def binary_conditions(x):
    hand_l_closed = get_percentage_armpose(x, 6)
    hand_r_closed = get_percentage_armpose(x, 7)
    arm_l_up = get_percentage_armpose(x, 4)
    arm_r_up = get_percentage_armpose(x, 5)
    return hand_r_closed, hand_l_closed, arm_r_up, arm_l_up

def repetitions(data, idx):
    d = [float(s[idx]) for s in data]
    middle = statistics.median(set(d))
    switches = 0
    above = None
    for val in d:
        if above is None:
            above = True if val > middle else False
        if val > middle:
            if above is False:
                switches += 1
        elif val < middle:
            if above is True:
                switches += 1
    switches = (switches -1) if switches > 0 else 0
    return switches

def conditions_dance(x, difficulty="normal"):
    b1, b2, b3, b4 = binary_conditions(x)
    d = {"normal": [20,10,-60,45, -60], "easy": [15,5,-65, 40, -55], "hard": [25,15,-55,50, -65]}
    d = d[difficulty]
    binary_ok = (b4 > 0.70 and b3 > 0.70) if len(x[0]) == 8 else True
    if all((stdev(x, 1) > d[0], stdev(x,3) > d[1], mean(x,2) < d[2], mean(x,0) > d[3], mean(x,0)>0, mean(x,1)<0, mean(x,2) < d[4], mean(x, 3 )>0)):
        if (repetitions(x,1) > 1 and repetitions(x, 3) > 1) or len(x) <= 5 and binary_ok:
            return True
    return False

def conditions_fly(x, difficulty="normal"):
    b1, b2, b3, b4 = binary_conditions(x)
    d = {"normal": [17,17,50,-50, 50, 60, -30, 0.50], "easy": [15,15,40,-65, 60, 65, -25, 0.35], "hard": [20,20,55,-40,40,55, -35, -0.65]}
    d = d[difficulty]
    binary_ok = (b4 < 0.30 and b3 < 0.30) if len(x[0]) == 8 else True
    if all((stdev(x, 0) > d[0], stdev(x,2) > d[1], mean(x,0)>0, d[3] < mean(x,1)<0, mean(x,2) < 0, d[4] > mean(x, 3 )>0)):
        if ((repetitions(x,0) > 1 and repetitions(x, 2) > 1) or len(x) <= 5) and binary_ok:
            return True
    return False

def conditions_wave(x, difficulty="normal"):
    b1, b2, b3, b4 = binary_conditions(x)
    d = {"normal": [17,12,15,25, 0.4], "easy": [20, 15, 10, 30, 0.25], "hard": [15, 10, 25, 20, 0.55]}
    d = d[difficulty]
    binary_ok = (b4 > 0.70 and b3 < 0.30) if len(x[0]) == 8 else True
    if all((stdev(x, 0) < d[0], stdev(x, 2) < d[1], stdev(x, 1) > d[2], mean(x, 3) < d[3], mean(x,0)>0, mean(x,1)<0, mean(x,2) < 0, mean(x, 3 )>0)):
        if (repetitions(x,1) > 1 or len(x) <= 5) and binary_ok:
            return True
    return False

def label_to_cls(model, label):
        classes_taken = list(model.classes.values())
        if label not in model.classes.keys():
            new_idx = 0
            while True:
                if new_idx not in classes_taken:
                    break
                else:
                    new_idx += 1
            model.classes.update({label:torch.nn.Parameter(torch.tensor(new_idx).float().cuda())})
        return model.classes[label]


def cls_to_label(model, cls):
    for v in model.classes.keys():
        if model.classes[v] == cls:
           return v
    return None

def check_action_ok(data, label, difficulty="normal"):
        success_matrix = []
        for idx, x in enumerate(data):
            x = np.asarray([y for y in x if sum(y) != 0])
            x = x * 180
            if label[idx] == "fly":
                if conditions_fly(x, difficulty):
                    success_matrix.append(1)
                else:
                    success_matrix.append(0)
            elif label[idx] == "wave":
                if conditions_wave(x, difficulty):
                        success_matrix.append(1)
                else:
                        success_matrix.append(0)
            elif label[idx] == "dance":
                if  conditions_dance(x, difficulty):
                    success_matrix.append(1)
                else:
                    success_matrix.append(0)
        return success_matrix

def load_images(path, dim):
    def generate(images):
        dataset = np.zeros((len(images), dim[0], dim[1], dim[2]), dtype=np.float)
        for i, image_path in enumerate(images):
            image = imageio.imread(image_path)
            dataset[i, :] = image / 255
        return dataset.reshape(-1, dataset.shape[-1], dataset.shape[1], dataset.shape[2])

    if any([os.path.isdir(x) for x in glob.glob(os.path.join(path, "*"))]):
        subparts = (glob.glob(os.path.join(path, "./*")))
        datasets = []
        for s in subparts:
            images = (glob.glob(os.path.join(s, "*.png")))
            d = generate(images)
            datasets.append(d)
        return np.concatenate(datasets)
    else:
        images = (glob.glob(os.path.join(path, "*.png")))
        dataset = generate(images)
        return dataset


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

class Constants(object):
    eta = 1e-6
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88  # largest cuda v s.t. exp(v) < inf
    logfloorc = -104  # smallest cuda v s.t. exp(v) > 0

class Logger(object):
    """Saves training progress into csv"""

    def __init__(self, path, mods):
        self.fields = ["Epoch", "Train Loss", "Test Loss", "Train KLD", "Test KLD", "Test Loss dance", "Test Loss wave", "Test Loss fly", "Log Likelihood"]
        self.path = path
        self.dic = {}
        for m in range(len(mods)):
            self.fields.append("Train Mod_{}".format(m))
            self.fields.append("Test Mod_{}".format(m))
        self.reset()

    def reset(self):
        with open(os.path.join(self.path, "loss.csv"), mode='w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fields)
            writer.writeheader()

    def update_train(self, val_d):
        self.dic = val_d

    def update(self, val_d):
        with open(os.path.join(self.path, "loss.csv"), mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fields)
            writer.writerow({**self.dic, **val_d})
        self.dic = {}


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print('====> [{}] Time: {:7.3f}s or {}'
              .format(self.name,
                      self.elapsed,
                      time.strftime("%H:%M:%S", self.elapsedH)))


# Functions
def save_vars(vs, filepath):
    """
    Saves variables to the given filepath in a safe manner.
    """
    if os.path.exists(filepath):
        shutil.copyfile(filepath, '{}.old'.format(filepath))
    torch.save(vs, filepath)


def save_model(model, filepath):
    """
    To load a saved model, simply use
    `model.load_state_dict(torch.load('path-to-saved-model'))`.
    """
    save_vars(model.state_dict(), filepath)
    if hasattr(model, 'vaes'):
        for vae in model.vaes:
            fdir, fext = os.path.splitext(filepath)
            save_vars(vae.state_dict(), fdir + '_' + vae.modelName + fext)


def get_mean(d, K=100):
    """
    Extract the `mean` parameter for given distribution.
    If attribute not available, estimate from samples.
    """
    try:
        mean = d.loc
    except NotImplementedError:
        print("could not get mean")
        samples = d.rsample(torch.Size([K]))
        mean = samples.mean(0)
    return mean


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


def kl_divergence(d1, d2, K=100):
    """Computes closed-form KL if available, else computes a MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)


def pdist(sample_1, sample_2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances. Code
    adapted from the torch-two-sample library (added batching).
    You can find the original implementation of this function here:
    https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/util.py

    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(batch_size, n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(batch_size, n_2, d)``.
    norm : float
        The l_p norm to be used.
    batched : bool
        whether data is batched

    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (batch_size, n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    if len(sample_1.shape) == 2:
        sample_1, sample_2 = sample_1.unsqueeze(0), sample_2.unsqueeze(0)
    B, n_1, n_2 = sample_1.size(0), sample_1.size(1), sample_2.size(1)
    norms_1 = torch.sum(sample_1 ** 2, dim=-1, keepdim=True)
    norms_2 = torch.sum(sample_2 ** 2, dim=-1, keepdim=True)
    norms = (norms_1.expand(B, n_1, n_2)
             + norms_2.transpose(1, 2).expand(B, n_1, n_2))
    distances_squared = norms - 2 * sample_1.matmul(sample_2.transpose(1, 2))
    return torch.sqrt(eps + torch.abs(distances_squared)).squeeze()  # batch x K x latent


def NN_lookup(emb_h, emb, data):
    indices = pdist(emb.to(emb_h.device), emb_h).argmin(dim=0)
    # indices = torch.tensor(cosine_similarity(emb, emb_h.cpu().numpy()).argmax(0)).to(emb_h.device).squeeze()
    return data[indices]


import math
import torch
from torch.optim.optimizer import Optimizer
from tabulate import tabulate

version_higher = (torch.__version__ >= "1.5.0")


class Adabelief(Optimizer):
    r"""Implements AdaBelief algorithm. Modified from Adam in PyTorch
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-16)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: True) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: True) If set as True, then perform the rectified
            update similar to RAdam
        degenerated_to_sgd (boolean, optional) (default:True) If set as True, then perform SGD update
            when variance of gradient is high
        print_change_log (boolean, optional) (default: True) If set as True, print the modifcation to
            default hyper-parameters
    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients, NeurIPS 2020
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay=0, amsgrad=False, weight_decouple=True, fixed_decay=False, rectify=True,
                 degenerated_to_sgd=True, print_change_log=True):

        # ------------------------------------------------------------------------------
        # Print modifications to default arguments
        if print_change_log:
            print(Fore.RED + 'Please check your arguments if you have upgraded adabelief-pytorch from version 0.0.5.')
            print(Fore.RED + 'Modifications to default arguments:')
            default_table = tabulate([
                ['adabelief-pytorch=0.0.5', '1e-8', 'False', 'False'],
                ['>=0.1.0 (Current 0.2.0)', '1e-16', 'True', 'True']],
                headers=['eps', 'weight_decouple', 'rectify'])
            print(Fore.RED + default_table)

            recommend_table = tabulate([
                ['Recommended eps = 1e-8', 'Recommended eps = 1e-16'],
            ],
                headers=['SGD better than Adam (e.g. CNN for Image Classification)',
                         'Adam better than SGD (e.g. Transformer, GAN)'])
            print(Fore.BLUE + recommend_table)

            print(Fore.BLUE + 'For a complete table of recommended hyperparameters, see')
            print(Fore.BLUE + 'https://github.com/juntang-zhuang/Adabelief-Optimizer')

            print(
                Fore.GREEN + 'You can disable the log message by setting "print_change_log = False", though it is recommended to keep as a reminder.')

            print(Style.RESET_ALL)
        # ------------------------------------------------------------------------------

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, buffer=[[None, None, None] for _ in range(10)])
        super(Adabelief, self).__init__(params, defaults)

        self.degenerated_to_sgd = degenerated_to_sgd
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print('Weight decoupling enabled in AdaBelief')
            if self.fixed_decay:
                print('Weight decay fixed')
        if self.rectify:
            print('Rectification enabled in AdaBelief')
        if amsgrad:
            print('AMSGrad enabled in AdaBelief')

    def __setstate__(self, state):
        super(Adabelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']

                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)

                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)

                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # cast data type
                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaBelief does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                            if version_higher else torch.zeros_like(p.data)

                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var.add_(group['eps']), out=max_exp_avg_var)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # update
                if not self.rectify:
                    # Default update
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                else:  # Rectified update, forked from RAdam
                    buffered = group['buffer'][int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        N_sma_max = 2 / (1 - beta2) - 1
                        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                        buffered[1] = N_sma

                        # more conservative since it's an approximated value
                        if N_sma >= 5:
                            step_size = math.sqrt(
                                (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                        elif self.degenerated_to_sgd:
                            step_size = 1.0 / (1 - beta1 ** state['step'])
                        else:
                            step_size = -1
                        buffered[2] = step_size

                    if N_sma >= 5:
                        denom = exp_avg_var.sqrt().add_(group['eps'])
                        p.data.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    elif step_size > 0:
                        p.data.add_(exp_avg, alpha=-step_size * group['lr'])

                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half()

        return loss