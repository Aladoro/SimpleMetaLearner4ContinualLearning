import re
import hydra
import os.path as osp
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize


def get_grad_norm_module_debug(module: nn.Module, mode=2):
    named_params = module.named_parameters()
    if type(mode) == int:
        norms = []
        for n, p in named_params:
            grad = p.grad
            if grad is not  None:
                norms.append(torch.norm(p.grad.detach(), p=mode))
        norm = torch.norm(torch.stack(norms), p=mode)
    elif (mode == 'max') or (mode == 'inf'):
        norms = []
        for n, p in named_params:
            grad = p.grad
            if grad is not None:
                norms.append(p.grad.detach().abs().max())
        norm = torch.max(torch.stack(norms))
    else:
        raise NotImplementedError
    return norm

def get_grad_norm_module(module: nn.Module, mode=2):
    params = module.parameters()
    if type(mode) == int:
        norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), p=mode)
                                       for p in params if p.requires_grad]), p=mode)
    elif (mode == 'max') or (mode == 'inf'):
        norm = torch.max(torch.stack([p.grad.detach().abs().max()
                                      for p in params if p.requires_grad]))
    else:
        raise NotImplementedError
    return norm

@torch.no_grad()
def get_wd_params(model: nn.Module, relevant_params=None):
    if relevant_params is not None:
        relevant_params = list(relevant_params)
    else:
        relevant_params = list(model.parameters())
    wd_params = list()
    for m in model.modules():
        if isinstance(
                m,
                (
                        nn.Linear,
                        nn.Conv1d,
                        nn.Conv2d,
                        nn.Conv3d,
                        nn.ConvTranspose1d,
                        nn.ConvTranspose2d,
                        nn.ConvTranspose3d,
                ),
        ):
            print('applying decay to module {}'.format(m))
            is_relevant = False
            for p in relevant_params:
                if p is m.weight:
                    is_relevant = True
            if is_relevant:
                wd_params.append(m.weight)
    no_wd_params = list()
    for p in relevant_params:
        p_in_wd = False
        for wd_p in wd_params:
            if p is wd_p:
                p_in_wd = True
        if not p_in_wd:
            no_wd_params.append(p)

    assert len(wd_params) + len(no_wd_params) == len(relevant_params), "Sanity check failed."
    return wd_params, no_wd_params


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def dequantize(ims):
    return (ims / 127.5 - 1)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def freeze_legacy_learner(model, total_frozen_layers=13):
    frozen_params_name = ['vars.{}'.format(i) for i in range(total_frozen_layers * 2)]
    for name, param in model.named_parameters():
        if name in frozen_params_name:
            print('Freezing {}'.format(name))
            param.requires_grad = False
    if model.Neuromodulation:
        reset_params_name = ['vars.26', 'vars.27']
    else:
        reset_params_name = ['vars.14', 'vars.15']
    for name, param in model.named_parameters():
        if name in reset_params_name:
            print('Resetting {}'.format(name))
            original_init(param)


def freeze_model_components(model, components=[], verbose=False):
    not_frozen_params = []
    for name, parameter in model.named_parameters():
        freeze = False
        for component in components:
            if component in name:
                freeze = True
        if freeze:
            if verbose:
                print('Freezing {}'.format(name))
            parameter.requires_grad = False
        else:
            if parameter.requires_grad:
                not_frozen_params.append(name)
    print('Not frozen, still optimized parameters:')
    for name in not_frozen_params:
        print(name)



def unfreeze_batchnorm_params(m):
    if isinstance(m, nn.BatchNorm2d):
        for p in m.parameters():
            p.requires_grad = True


def weight_init(m, c=None):
    if c is not None:
        gain = nn.init.calculate_gain('relu')
        assert isinstance(m, nn.Linear)
        nn.init.orthogonal_(m.weight.data[c].unsqueeze(0), gain)
        if hasattr(m.bias, 'data'):
            m.bias.data[c].fill_(0.0)
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def final_weight_init(m):
    nn.init.orthogonal_(m.weight.data, gain=0.01)
    if hasattr(m.bias, 'data'):
        m.bias.data.fill_(0.0)


def original_init(m, c=None):
    if c is not None:
        assert isinstance(m, nn.Linear)
        reset_weight = torch.nn.init.kaiming_normal_(
            torch.zeros_like(m.weight.data[c].unsqueeze(0)))  # now same as impl
        m.weight.data[c] = reset_weight

    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, torch.Tensor):
        dimensions = len(m.size())
        if dimensions == 1:
            m.data = torch.zeros_like(m)
        elif dimensions == 2 or dimensions == 4:
            reset_weight = torch.nn.init.kaiming_normal_(
                torch.zeros_like(m))
            m.data = reset_weight

def add_new_attribute(module, attribute, value=None):
    if hasattr(module, attribute):
        pass
    else:
        print('adding {}=None'.format(attribute))
        setattr(module, attribute, value)

def make_learner_model_compatible(model):
    add_new_attribute(module=model, attribute='meta_optim_counter', value=0)

def reinit_model_components(model, components=[], reinit_fn=original_init,
                            reinit_memory=True, verbose=False):
    for name, parameter in model.named_parameters():
        reinit = False
        for component in components:
            if component in name:
                reinit = True
        if reinit:
            if verbose:
                print('Resetting {}'.format(name))
            reinit_fn(parameter)
    if reinit_memory:
        if hasattr(model, 'memory'):
            model.memory.latent_index = 0
            model.memory.full = False
            if verbose:
                print('Resetting memory index')
    make_learner_model_compatible(model=model)
    if getattr(model, 'q_condition_permute_features_prob', None) is not None:
        pass
    else:
        model.q_condition_permute_features_prob = 0
    if getattr(model, 'mask_current_labels_ltm_prob', None) is not None:
        if model.mask_current_labels_ltm_prob < 1:
            print('Setting current labels ltm masking prob to 0')
            model.mask_current_labels_ltm_prob = 0

    if getattr(model, 'reset_wm_prob', None) is not None:
        print('Setting reset WM prob to 0')
        if model.reset_wm_prob < 1:
            model.reset_wm_prob = 0
    else:
        model.reset_wm_prob = 0
    if getattr(model, 'mask_short_term_prob', None) is not None:
        if model.mask_short_term_prob < 1:
            print('Setting mask short term prob to 0')
            model.mask_short_term_prob = 0
    else:
        model.mask_short_term_prob = 0
    if getattr(model, 'mask_long_term_prob', None) is not None:
        if model.mask_long_term_prob < 1:
            print('Setting mask long term prob to 0')
            model.mask_long_term_prob = 0
    else:
        model.mask_long_term_prob = 0

    if getattr(model, 'mask_wm_prob', None) is not None:
        if model.mask_wm_prob < 1:
            print('Setting mask wm prob to 0')
            model.mask_wm_prob = 0
    else:
        model.mask_wm_prob = 0


def iterator_learner_statistics(learner, iterator, device='cuda', learner_kwargs=None,
                                max_class=-1, ordered_data=False):
    if learner_kwargs is None:
        learner_kwargs = {}
    accuracies = []
    total_correct = 0
    total_evaluated = 0
    unseen_samples = False
    with torch.no_grad():
        for data, labels in iterator:
            data, labels = data.to(device), labels.to(device)
            if max_class >= 0:
                valid_samples = labels <= max_class
                mask = valid_samples.to(dtype=torch.int32)
                total_evaluated += mask.sum()
                if torch.all(valid_samples):
                    pass
                else:
                    unseen_samples = True
            else:
                mask = 1.0
                total_evaluated += data.size()[0]
            labels = labels.to(device)
            logits = learner(data, **learner_kwargs)
            classes = logits.argmax(dim=1)
            total_correct += (torch.eq(classes, labels).to(dtype=torch.int32) * mask).sum()
            if unseen_samples and ordered_data:
                break
    return (total_correct / total_evaluated).item()


def iterator_trainer_statistics(trainer, iterator):
    accuracies = []
    losses = []
    for data, labels in iterator:
        with torch.no_grad():
            data, labels = data.to(trainer.device), labels.to(trainer.device)
            labels = labels.to(trainer.device)
            loss, logits, _ = trainer.loss(data=data, labels=labels, inner_params=None)
            classes = logits.argmax(dim=1)
            accuracies.append(torch.eq(classes, labels).to(dtype=torch.float32).mean().item())
            losses.append(torch.mean(loss).item())
    return {'mean_accuracy': np.mean(accuracies), 'mean_loss': np.mean(losses)}


def load_config(experiment_path):
    hydra_folder = osp.join(experiment_path, ".hydra")
    with initialize(config_path=hydra_folder):
        cfg = compose(config_name="config")
    return cfg


def load_learner_cfg(experiment_path, model_name=None, iteration=None):
    cfg = load_config(experiment_path=experiment_path)
    learner = hydra.utils.instantiate(cfg.learner)
    if model_name is None:
        if iteration is not None:
            model_name = "learner_{}.net".format(iteration)
        else:
            print('WARNING: loading randomly initialized learner')
            return learner, cfg
    model_path = osp.join(experiment_path, model_name)
    learner = torch.load(model_path)
    return learner, cfg


def soft_cross_entropy(logits, soft_labels, reduce='mean'):
    log_probs = F.log_softmax(input=logits, dim=-1)
    losses = -1 * (log_probs * soft_labels).sum(-1)
    if reduce == 'None':
        return losses
    elif reduce == 'mean':
        return losses.mean()
    elif reduce == 'mean':
        return losses.mean()
    elif reduce == 'sum':
        return losses.sum()
    else:
        raise NotImplementedError


def setup_optimizer(model, optimizer_cfg):
    if hasattr(model, 'not_regularized_params'):
        assert hasattr(model, 'regularized_params')
        reg_params = model.regularized_params()
        nreg_params = model.not_regularized_params()
        assert len(list(model.parameters())) == len(reg_params) + len(nreg_params)
        print('Omitting {}/{} parameters from regularization'.format(
            len(nreg_params), len(reg_params) + len(nreg_params)))
        optimizer = hydra.utils.instantiate(optimizer_cfg, params=iter([{'params': reg_params, },
                                                                        {'params': nreg_params, 'weight_decay': 0.0}]),
                                            _recursive_=False,)
    else:
        print('Regularizing all model parameters')
        optimizer = hydra.utils.instantiate(optimizer_cfg, params=model.parameters())
    return optimizer
