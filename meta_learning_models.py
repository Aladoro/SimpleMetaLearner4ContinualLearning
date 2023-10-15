from typing import Optional

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from training_utils import get_wd_params, get_grad_norm_module
from logging_utils import Metrics
import numpy as np


class MetaLearningClassifier(nn.Module):
    def __init__(self,
                 device,
                 learner,
                 inner_updates_lr,
                 inner_update_steps,
                 meta_aggregate_task_data,
                 meta_reset_class_weights,
                 inner_total_samples=None,
                 meta_aggregate_unused_task_data=True,
                 meta_optim_name='Adam',
                 meta_lr=1e-3,
                 meta_optim_kwargs={},
                 meta_lr_scheduler=None,
                 meta_lr_scheduler_kwargs={},
                 meta_reset_random_classes=0,
                 meta_loss='fomaml',
                 meta_optim_use_only_inner_losses=False,
                 meta_inner_losses_coeff=0,
                 preserve_inner_changes=False,
                 collect_meta_loss_every=None,
                 current_task_meta_loss_coeff=None,
                 all_tasks_meta_loss_coeff=None,
                 record_learned_classes=False,
                 meta_weight_decay=None,
                 keep_indexing_first_step_optim=False,
                 meta_only_use_task_data=False,
                 ):
        # learning_rate_modulator=None): in learner for eval

        super(MetaLearningClassifier, self).__init__()
        self.meta_only_use_task_data = meta_only_use_task_data
        self.idx_b=False
        self.keep_indexing_first_step_optim = keep_indexing_first_step_optim
        self.device = device
        # inner/outer lr & n_inner_updates
        self.inner_updates_lr = inner_updates_lr
        self.inner_update_steps = inner_update_steps
        self.inner_batch_size = 1

        if inner_total_samples is None:
            self.inner_total_samples = self.inner_update_steps
        else: # TODO
            assert inner_total_samples >= self.inner_update_steps
            self.inner_total_samples = inner_total_samples
            if self.inner_update_steps < inner_total_samples:
                assert (meta_aggregate_task_data or meta_aggregate_unused_task_data) == True, 'If not reused to aggregate task data, ' \
                                                         'extra samples would be wasted'
        self.preserve_inner_changes = preserve_inner_changes
        self.meta_lr = meta_lr
        self.meta_aggregate_task_data = meta_aggregate_task_data
        self.meta_aggregate_unused_task_data = meta_aggregate_unused_task_data
        self.unused_task_samples = self.inner_total_samples - self.inner_update_steps


        self.meta_optim_use_only_inner_losses = meta_optim_use_only_inner_losses
        if self.meta_optim_use_only_inner_losses:
            print(self.meta_aggregate_task_data)
            assert meta_aggregate_task_data == False
            assert meta_inner_losses_coeff == 0
            assert meta_aggregate_unused_task_data #
        self.meta_inner_losses_coeff = meta_inner_losses_coeff

        self.meta_reset_class_weights = meta_reset_class_weights
        self.meta_reset_random_classes = meta_reset_random_classes

        if collect_meta_loss_every == None:
            self.collect_meta_loss_every = self.inner_update_steps
        else:
            assert self.inner_update_steps % collect_meta_loss_every == 0
            self.collect_meta_loss_every = collect_meta_loss_every
        self.num_classes = 1000
        self.record_learned_classes = record_learned_classes
        if self.record_learned_classes:
            self.learned_classes_vector = torch.zeros([self.num_classes], device=self.device)
        else:
            self.learned_classes_vector = None
        self.learner = learner
        meta_learned_parameters = list(self.learner.parameters())
        if self.preserve_inner_changes:
            all_parameters = self.learner.parameters()
            meta_learned_parameters = []
            for p in all_parameters:
                is_inner = False
                for ip in self.learner.inner_params:
                    if p is ip:
                        is_inner = True
                if not is_inner:
                    meta_learned_parameters.append(p)

        if meta_optim_name == 'Adam':
            torch_optim = torch.optim.Adam
        else:
            raise NotImplementedError

        if meta_weight_decay:
            wd_params, nwd_params = get_wd_params(self.learner,
                                                  relevant_params=meta_learned_parameters)

            optim_groups = [
                {"params": wd_params, "weight_decay": meta_weight_decay},
                {"params": nwd_params, "weight_decay": 0.0},
            ]
            self.meta_optimizer = torch_optim(optim_groups, lr=self.meta_lr, **meta_optim_kwargs)
        else:
            self.meta_optimizer = torch_optim(meta_learned_parameters,
                                             lr=self.meta_lr, **meta_optim_kwargs)
        if meta_lr_scheduler is not None:
            if meta_lr_scheduler == 'cos' or meta_lr_scheduler == 'cosine':
                assert 'T_max' in meta_lr_scheduler_kwargs
                assert 'eta_min' in meta_lr_scheduler_kwargs
                self.meta_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=self.meta_optimizer, **meta_lr_scheduler_kwargs)
            elif meta_lr_scheduler == 'linear':
                assert 'start_factor' in meta_lr_scheduler_kwargs
                assert 'end_factor' in meta_lr_scheduler_kwargs
                assert 'total_iters' in meta_lr_scheduler_kwargs
                self.meta_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer=self.meta_optimizer, **meta_lr_scheduler_kwargs)
            else:
                raise NotImplementedError
        else:
            self.meta_lr_scheduler = None

        self.trainer_logging_metrics = Metrics()
        self.current_task_meta_loss_coeff = current_task_meta_loss_coeff
        self.all_tasks_meta_loss_coeff = all_tasks_meta_loss_coeff
        if self.current_task_meta_loss_coeff is not None or self.all_tasks_meta_loss_coeff is not None:
            self.trainer_logging_metrics.add('all_tasks_meta_loss', 'current_tasks_meta_loss')
            if self.current_task_meta_loss_coeff is None:
                self.current_task_meta_loss_coeff = 1
            if self.all_tasks_meta_loss_coeff is None:
                self.all_tasks_meta_loss_coeff = 1
        self.meta_loss = meta_loss
        self.device = None
        self.meta_iteration = 0

    def sample_training_data(self, task_iterators, validation_task_iterators,
                             all_tasks_iterators, total_task_samples,
                             all_tasks_buffer=None, ):
        meta_batch_size = len(all_tasks_iterators)
        total_task_iterators = len(task_iterators)
        assert total_task_iterators % meta_batch_size == 0

        iterators_per_meta_sample = total_task_iterators // meta_batch_size
        task_batch_size = task_iterators[0].batch_size
        if all_tasks_iterators is not None:
            all_tasks_batch_size = all_tasks_iterators[0].batch_size
        elif all_tasks_buffer is not None:
            assert meta_batch_size == 1
            all_tasks_batch_size = all_tasks_buffer.default_input_data_batch_size
        else:
            raise NotImplementedError
        if validation_task_iterators is not None:
            validation_task_batch_size = validation_task_iterators[0].batch_size
        else:
            validation_task_batch_size = 0
        meta_batch_all_tasks_indices = all_tasks_batch_size

        # for every meta batch size we are adding validation data from iterators_per_meta_sample tasks
        meta_batch_current_task_indices = validation_task_batch_size * iterators_per_meta_sample

        assert total_task_samples % iterators_per_meta_sample == 0
        steps_per_iterator = total_task_samples // iterators_per_meta_sample
        assert steps_per_iterator % task_batch_size == 0

        meta_batch_inner_data = []
        meta_batch_inner_labels = []
        meta_batch_all_tasks_data = []
        meta_batch_all_tasks_labels = []

        for i in range(meta_batch_size):
            inner_data = []
            inner_labels = []
            meta_sample_task_iterators = task_iterators[
                i * iterators_per_meta_sample:(i + 1) * iterators_per_meta_sample]
            if all_tasks_iterators is not None:
                all_tasks_data, all_tasks_labels = next(iter(all_tasks_iterators[i]))
            elif all_tasks_buffer is not None:
                all_tasks_data, all_tasks_labels = all_tasks_buffer.sample_inputs()
            else:
                raise NotImplementedError

            for j, it in enumerate(meta_sample_task_iterators):
                it = iter(it)
                for i in range(0, steps_per_iterator, task_batch_size):
                    data, label = next(it)
                    inner_data.append(data)
                    inner_labels.append(label)
                if validation_task_iterators is not None:
                    validation_task_data, validation_task_labels = next(iter(validation_task_iterators[j]))
                    all_tasks_data = torch.cat([all_tasks_data, validation_task_data], dim=0)
                    all_tasks_labels = torch.cat([all_tasks_labels, validation_task_labels], dim=0)

            inner_data = torch.cat(inner_data, dim=0)
            inner_labels = torch.cat(inner_labels, dim=0)

            meta_batch_inner_data.append(inner_data)
            meta_batch_inner_labels.append(inner_labels)

            meta_batch_all_tasks_data.append(all_tasks_data)
            meta_batch_all_tasks_labels.append(all_tasks_labels)

        meta_batch_inner_data = torch.stack(meta_batch_inner_data, dim=0).to(self.device)
        meta_batch_inner_labels = torch.stack(meta_batch_inner_labels, dim=0).to(self.device)
        meta_batch_all_tasks_data = torch.stack(meta_batch_all_tasks_data, dim=0).to(self.device)
        meta_batch_all_tasks_labels = torch.stack(meta_batch_all_tasks_labels, dim=0).to(self.device)

        return (meta_batch_inner_data, meta_batch_inner_labels, meta_batch_all_tasks_data, meta_batch_all_tasks_labels,
                meta_batch_all_tasks_indices, meta_batch_current_task_indices)

    def inner_update(self, data, labels, inner_params=None, force_fomaml=False):
        if force_fomaml:
            meta_loss = 'fomaml'
        else:
            meta_loss = self.meta_loss
        inner_params, new_sample_loss, logits = self.learner.inner_update(
            data, labels, step_lr=self.inner_updates_lr,
            meta_loss=meta_loss, inner_params=inner_params)
        return inner_params, new_sample_loss, logits

    def loss(self, data, labels, inner_params):
        return self.learner.loss(data=data, labels=labels, inner_params=inner_params,
                                 inner_batch_size=self.inner_batch_size)

    def meta_step(self, inner_data,
                  inner_labels,
                  inner_update_steps,
                  all_tasks_data, all_tasks_labels,
                  meta_batch_all_tasks_indices,
                  meta_batch_current_task_indices,
                  ):
        inner_params = None
        inner_data_list = torch.split(
            inner_data[:inner_update_steps],
            split_size_or_sections=self.inner_batch_size,
            dim=0)

        inner_labels_list = torch.split(
            inner_labels[:inner_update_steps],
            split_size_or_sections=self.inner_batch_size,
            dim=0)
        meta_step_losses = []
        meta_step_logits = []

        step_data_losses = []
        latest_step_data_losses = []
        step_data_logits = []
        latest_step_data_logits = []

        for i, (step_data, step_labels) in enumerate(zip(inner_data_list, inner_labels_list)):
            inner_params, step_loss, step_logits = self.inner_update(data=step_data,
                                                                     labels=step_labels,
                                                                     inner_params=inner_params)
            if self.record_learned_classes:
                if self.learned_classes_vector[step_labels] == 0:
                    step_loss = torch.zeros_like(step_loss)
                self.learned_classes_vector[step_labels] = 1.0


            step_data_losses.append(step_loss)
            step_data_logits.append(step_logits)
            latest_step_data_logits.append(step_logits)
            latest_step_data_losses.append(step_loss)
            check_index = i + 1
            if self.idx_b:
                check_index = i
            if check_index % self.collect_meta_loss_every == 0 or ((i == 0) and self.keep_indexing_first_step_optim):
                if self.current_task_meta_loss_coeff is not None:
                    meta_loss_all_tasks_data, meta_loss_current_tasks_data = torch.split(
                        all_tasks_data, split_size_or_sections=[meta_batch_all_tasks_indices,
                                                                meta_batch_current_task_indices],
                        dim=0)
                    meta_loss_all_tasks_labels, meta_loss_current_tasks_labels = torch.split(
                        all_tasks_labels, split_size_or_sections=[meta_batch_all_tasks_indices,
                                                                  meta_batch_current_task_indices],
                        dim=0)
                    meta_loss_all_tasks, logits_all_tasks, _ = self.learner.loss(data=meta_loss_all_tasks_data,
                                                                                 labels=meta_loss_all_tasks_labels,
                                                                                 inner_params=inner_params,
                                                                                 learned_classes=self.learned_classes_vector)

                    if self.meta_optim_use_only_inner_losses:
                        assert meta_loss_current_tasks_data.shape[0] == 0
                        meta_loss_current_tasks, logits_current_tasks = torch.stack(latest_step_data_losses, dim=0).mean(), step_data_logits
                    else:
                        meta_loss_current_tasks, logits_current_tasks, _ = self.learner.loss(
                            data=meta_loss_current_tasks_data, labels=meta_loss_current_tasks_labels,
                            inner_params=inner_params, learned_classes=self.learned_classes_vector)

                    self.trainer_logging_metrics.update(all_tasks_meta_loss=meta_loss_all_tasks.item(),
                                                        current_tasks_meta_loss=meta_loss_current_tasks.item())

                    meta_loss = (
                                self.all_tasks_meta_loss_coeff * meta_loss_all_tasks + self.current_task_meta_loss_coeff * meta_loss_current_tasks)
                    logits = torch.concat([logits_all_tasks, logits_current_tasks], dim=0)

                else:
                    meta_loss, logits, _ = self.learner.loss(data=all_tasks_data, labels=all_tasks_labels,
                                                             inner_params=inner_params,
                                                             learned_classes=self.learned_classes_vector)
                    if self.meta_optim_use_only_inner_losses:
                        step_data_samples = torch.concat(latest_step_data_logits, dim=0).shape[0]
                        all_tasks_samples = logits.shape[0]
                        total_samples = step_data_samples + all_tasks_samples
                        mean_step_data_loss = torch.stack(latest_step_data_losses, dim=0).mean()
                        meta_loss = meta_loss * all_tasks_samples/total_samples + mean_step_data_loss * step_data_samples/all_tasks_samples
                meta_step_losses.append(meta_loss)
                meta_step_logits.append(logits)
                latest_step_data_losses, latest_step_data_logits = [], []
        return meta_step_losses, meta_step_logits, step_data_losses, step_data_logits, inner_params

    def meta_train(self, task_iterators, validation_task_iterators,
                   all_tasks_iterators, inner_update_steps=None):

        if inner_update_steps == None:
            inner_update_steps = self.inner_update_steps
        meta_batch_size = len(all_tasks_iterators)
        (meta_batch_inner_data, meta_batch_inner_labels,
         meta_batch_all_tasks_data, meta_batch_all_tasks_labels,
         meta_batch_all_tasks_indices, meta_batch_current_task_indices) = (
            self.sample_training_data(task_iterators=task_iterators,
                                      all_tasks_iterators=all_tasks_iterators,
                                      validation_task_iterators=validation_task_iterators,
                                      total_task_samples=self.inner_total_samples, ))  # self.inner_update_steps, ))
        if self.meta_only_use_task_data:
            meta_batch_all_tasks_data = meta_batch_inner_data
            meta_batch_all_tasks_labels =  meta_batch_inner_labels
        elif self.meta_aggregate_task_data:
            meta_batch_all_tasks_data = torch.cat([meta_batch_all_tasks_data, meta_batch_inner_data], dim=1)
            meta_batch_all_tasks_labels = torch.cat([meta_batch_all_tasks_labels, meta_batch_inner_labels], dim=1)
            # inner update steps is already the total number of steps.
            meta_batch_current_task_indices += self.inner_total_samples
        elif self.meta_aggregate_unused_task_data and self.unused_task_samples > 0:
            # first dimension is meta_batch_size
            meta_batch_inner_data, meta_batch_inner_unused_data = torch.split(
                meta_batch_inner_data,
                split_size_or_sections=[self.inner_update_steps, self.unused_task_samples],
                dim=1)
            meta_batch_inner_labels, meta_batch_inner_unused_labels = torch.split(
                meta_batch_inner_labels,
                split_size_or_sections=[self.inner_update_steps, self.unused_task_samples],
                dim=1)
            meta_batch_all_tasks_data = torch.cat([meta_batch_all_tasks_data, meta_batch_inner_unused_data], dim=1)
            meta_batch_all_tasks_labels = torch.cat([meta_batch_all_tasks_labels, meta_batch_inner_unused_labels], dim=1)
            # inner update steps is already the total number of steps.
            meta_batch_current_task_indices += self.unused_task_samples

        if self.meta_reset_random_classes > 0:
            with torch.no_grad():
                classes = torch.randint(low=0, high=self.num_classes, size=[self.meta_reset_random_classes])
                for label in torch.split(classes, split_size_or_sections=1):
                    self.learner.reset_class_weights(c=label)
                    if self.record_learned_classes:
                        self.learned_classes_vector[label] = 0
        if self.meta_reset_class_weights:
            with torch.no_grad():
                unique_labels = torch.unique(meta_batch_inner_labels, sorted=False)
                for label in torch.split(unique_labels, split_size_or_sections=1):
                    self.learner.reset_class_weights(c=label)
                    if self.record_learned_classes:
                        self.learned_classes_vector[label] = 0

        meta_batch_losses, meta_batch_logits, meta_batch_step_data_losses = [], [], []
        for i in range(meta_batch_size):
            inner_data, inner_labels, all_tasks_data, all_tasks_labels = (
                meta_batch_inner_data[i], meta_batch_inner_labels[i],
                meta_batch_all_tasks_data[i], meta_batch_all_tasks_labels[i])
            (meta_step_losses, meta_step_logits, step_data_losses,
             step_data_logits, inner_params) = self.meta_step(
                inner_data=inner_data,
                inner_labels=inner_labels,
                inner_update_steps=inner_update_steps,
                all_tasks_data=all_tasks_data,
                all_tasks_labels=all_tasks_labels,
                meta_batch_all_tasks_indices=meta_batch_all_tasks_indices,
                meta_batch_current_task_indices=meta_batch_current_task_indices,
            )
            meta_batch_losses += meta_step_losses
            meta_batch_logits += torch.stack(meta_step_logits, dim=0)
            meta_batch_step_data_losses += step_data_losses

        with torch.no_grad():
            meta_batch_logits = torch.stack(meta_batch_logits, dim=0)
            classification_accuracy = torch.eq(meta_batch_logits.argmax(dim=-1),
                                               meta_batch_all_tasks_labels.unsqueeze(-2)).to(
                dtype=torch.float32).mean().item()
        meta_loss = torch.stack(meta_batch_losses, dim=0).mean()

        if self.meta_inner_losses_coeff > 0.0:  # todo examine, implement pot replacement
            average_meta_inner_loss = torch.stack(meta_batch_step_data_losses, dim=0).mean()
            meta_loss = meta_loss + self.meta_inner_losses_coeff * average_meta_inner_loss

        self.learner.zero_grad()
        meta_loss.backward()
        self.learner.process_meta_optim_grads()
        self.meta_optimizer.step()
        if self.preserve_inner_changes:
            with torch.no_grad():
                self.learner.update_inner_params(inner_params)
        self.learner.complete_meta_optim()
        self.meta_iteration += 1
        return classification_accuracy, meta_loss.item()

    def set_device(self, device):
        self.to(device=device)
        self.device = str(device)

    def logging_stats(self, ):
        logging_dict = self.learner.logging_stats()
        logging_dict.update(self.trainer_logging_metrics.get())
        if self.meta_lr_scheduler is not None:
            logging_dict['meta_lr'] = float(self.meta_lr_scheduler.get_last_lr()[0])
        if self.record_learned_classes:
            logging_dict['learned_classes'] = self.learned_classes_vector.sum().item()
        self.trainer_logging_metrics.reset()
        return logging_dict
