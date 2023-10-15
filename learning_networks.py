import logging
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from training_utils import original_init
from overridable_layers import OverLayer
from collections import OrderedDict
from logging_utils import Metrics
from training_utils import get_grad_norm_module
import hydra


def run_over_layers(input, layers, over_params=None):
    out = input
    over_counter = 0
    for i, l in enumerate(layers):
        if over_params is not None and isinstance(l, OverLayer):
            out = l(out, over_params[over_counter:over_counter + l.over_params])
            over_counter += l.over_params
        else:
            out = l(out)
    return out


def get_over_params(layers):
    params = []
    counter = 0
    for layer in layers:
        print(layer)
        if isinstance(layer, OverLayer):
            layer_over_params = list(layer.parameters())
            assert len(layer_over_params) == layer.over_params
            params += layer_over_params
            counter += layer.over_params
    print(counter)
    return params, counter


def check_over_layers(layers):
    return any(isinstance(l, OverLayer) for l in layers)


class OriginalLearningNetwork(nn.Module):
    def __init__(self, networks, neuromodulation=True, init_fn=original_init,
                 learning_rate_modulator=None,
                 log_meta_gradient_every=-1,):
        super(OriginalLearningNetwork, self).__init__()
        self.rln_layers, self.pln_layers, *other_layers = networks
        if neuromodulation:
            self.neuromod_layers, *other_layers = other_layers

        self.neuromodulation = neuromodulation
        self.init_fn = init_fn

        self.pln_params = list(self.pln_layers.parameters())
        self.rln_params = list(self.rln_layers.parameters())

        self.inner_params = []
        self.inner_params_sizes = {}
        if check_over_layers(self.rln_layers):
            rln_over_params, n_rln_over_params = get_over_params(self.rln_layers)
            self.inner_params += rln_over_params
            self.inner_params_sizes['rln_params'] = n_rln_over_params
        if check_over_layers(self.pln_layers):
            pln_over_params, n_pln_over_params = get_over_params(self.pln_layers)
            self.inner_params += pln_over_params
            self.inner_params_sizes['pln_params'] = n_pln_over_params
        if init_fn:
            self.apply(init_fn)

        self.named_inner_params = OrderedDict()
        self.inner_params_names = []

        for ip in self.inner_params:
            for n, p in self.named_parameters():
                if ip is p:
                    self.named_inner_params[n] = p
                    self.inner_params_names.append(n)
        named_inner_params_check = [any([p is nip for k, nip in self.named_inner_params.items()])
                                    for p in self.inner_params]
        self.learner_logging_metrics = Metrics()
        self.log_meta_gradient_every = log_meta_gradient_every
        self.meta_optim_counter = 0

        assert all(named_inner_params_check)

        self.learning_rate_modulator = learning_rate_modulator
        if self.learning_rate_modulator:
            self.learning_rate_modulator.add_indexed_parameters(
                indexed_parameters=self.named_inner_params)


    def training_loss(self, data, labels, inner_params, learned_classes=None,
                      loss_kwargs={},
                      **kwargs):
        logits = self.__call__(data, inner_params=inner_params)
        if learned_classes is not None:
            per_sample_loss = F.cross_entropy(logits, labels, reduction='none', **loss_kwargs)
            valid = learned_classes[labels]  # bs
            valid_loss = per_sample_loss * valid
            loss = valid_loss.sum() / (valid.sum() + 1e-8)
        else:
            loss = F.cross_entropy(logits, labels, **loss_kwargs)
        return loss, logits, loss

    def loss(self, data, labels, inner_params, learned_classes=None,
             loss_kwargs={}, **kwargs):
        return self.training_loss(data=data, labels=labels,
                                  inner_params=inner_params,
                                  learned_classes=learned_classes,
                                  **loss_kwargs)

    def split_inner_params(self, inner_params):
        rln_params, pln_params = None, None
        rln_size = self.inner_params_sizes.get('rln_params', 0)
        if rln_size > 0:
            rln_params = inner_params[:rln_size]
        pln_size = self.inner_params_sizes.get('pln_params', 0)
        if pln_size > 0:
            pln_params = inner_params[rln_size:rln_size + pln_size]
        return rln_params, pln_params

    def forward(self, input, rln_params=None, pln_params=None,
                inner_params=None, neuromod_params=None,
                return_latents=False):
        if inner_params is not None:
            rln_params, pln_params = self.split_inner_params(inner_params=inner_params)
        rln_out = run_over_layers(input=input, layers=self.rln_layers,
                                  over_params=rln_params)
        if self.neuromodulation:
            nm_out = run_over_layers(input=input, layers=self.neuromod_layers,
                                     over_params=neuromod_params)
            rln_out = rln_out * nm_out

        pln_out = run_over_layers(input=rln_out, layers=self.pln_layers,
                                  over_params=pln_params)
        if return_latents:
            return pln_out, rln_out
        else:
            return pln_out

    def zero_grad(self, params=None):
        with torch.no_grad():
            if params is None:
                for p in self.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()

    def inner_update(self, data, labels, step_lr, meta_loss, inner_params=None):
        if inner_params is None:
            inner_params = [torch.clone(p) for p in self.inner_params]
        loss, logits, new_sample_loss = self.training_loss(data=data, labels=labels, inner_params=inner_params)
        if meta_loss == 'fomaml':
            inner_grads = torch.autograd.grad(loss, inner_params, allow_unused=False)
        elif meta_loss == 'maml':
            inner_grads = torch.autograd.grad(loss, inner_params, allow_unused=False, create_graph=True)
        else:
            raise NotImplementedError
        if self.learning_rate_modulator:
            updated_inner_params = []
            for param, param_name, grad in zip(inner_params, self.inner_params_names, inner_grads):
                updated_param = param - self.learning_rate_modulator.get_lr(
                    param_name) * self.inner_updates_lr * grad
                updated_inner_params.append(updated_param)
            inner_params = updated_inner_params
        else:
            inner_params = [param - step_lr * grad for param, grad in zip(inner_params, inner_grads)]
        return inner_params, new_sample_loss, logits

    def reset_class_weights(self, c):
        self.init_fn(self.pln_layers[-1], c=c)

    def update_inner_params(self, new_inner_params):
        for param, target_param in zip(self.inner_params, new_inner_params):
            param.data.copy_(target_param)

    def logging_stats(self, ):
        logging_dict = dict()
        logging_dict.update({k: v.item() for k, v in self.learner_logging_metrics.get().items()})
        self.learner_logging_metrics.reset()
        logging_dict['meta_optim_steps'] = self.meta_optim_counter
        if self.learning_rate_modulator:
            modulator_state = self.learning_rate_modulator.get_state()
            for k, v in modulator_state.items():
                logging_dict['modulator_{}'.format(k)] = v.item()
        return logging_dict

    def process_meta_optim_grads(self,):
        self.meta_optim_counter = self.meta_optim_counter + 1

    def complete_meta_optim(self, ):
        pass
