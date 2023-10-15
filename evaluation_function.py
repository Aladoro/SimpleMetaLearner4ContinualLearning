import time
import torch
import hydra
import numpy as np
import os
from omegaconf import OmegaConf
import os.path as osp
import datasets.datasetfactory as df
from datasets.utils import prepare_dataset_iterators
from datasets.task_sampler import get_sampler
from training_utils import freeze_model_components, reinit_model_components, \
    original_init, unfreeze_batchnorm_params, freeze_legacy_learner
import torch.nn.functional as F

from pathlib import Path

from logging_utils import Metrics, setup_loggers, log_dict
from training_utils import iterator_learner_statistics
from hydra import compose, initialize


def evaluate_iterators(learner, evaluation_iterators={}, learner_kwargs=None,
                       device='cuda', max_class=-1, ordered_data=False):
    results_dict = {}
    for iterator_label, iterator in evaluation_iterators.items():
        accuracy = iterator_learner_statistics(
            learner=learner, iterator=iterator, device=device,
            learner_kwargs=learner_kwargs, max_class=max_class,
            ordered_data=ordered_data)
        results_dict['{}_accuracy'.format(iterator_label)] = accuracy
    return results_dict


def learn_continual_classifier(learner, optimizer, training_iterator,
                               evaluation_iterators={}, evaluation_logger=None,
                               epochs=1, device='cuda', learner_kwargs=None,
                               log_eval_every=-1, log_eval_steps=[],
                               ordered_evaluation_data=False,
                               log_logging_stats=False,
                               print_wm=False):
    if learner_kwargs is None:
        learner_kwargs = {'inner_params': None}
    else:
        learner_kwargs['inner_params'] = None

    if log_eval_every > 0:
        assert evaluation_logger is not None

        def log_step(step, seen_classes):
            evaluation_logger.log_key_val('step', step)
            start_evaluation_time = time.time()
            if seen_classes is not None:
                max_class = seen_classes.max()
                num_seen = seen_classes.size()[0]
            else:
                max_class = -1
                num_seen = 0
            evaluation_logger.log_key_val('seen_classes', num_seen)
            results_dict = evaluate_iterators(learner=learner,
                                              evaluation_iterators=evaluation_iterators,
                                              learner_kwargs=learner_kwargs, device=device,
                                              max_class=max_class,
                                              ordered_data=ordered_evaluation_data)
            if log_logging_stats:
                results_dict.update(learner.logging_stats())
            log_dict(logger=evaluation_logger, dictionary=results_dict)
            evaluation_time = time.time() - start_evaluation_time
            evaluation_logger.log_key_val('evaluation_time', evaluation_time)
            evaluation_logger.log_iteration()
            results_dict['seen_classes'] = num_seen
            return results_dict
    else:
        def log_step(step):
            pass

    learner.train()
    current_step = 0
    last_eval_step = -1
    seen_classes = None
    all_results = None
    for epoch in range(epochs):
        for i, (data, labels) in enumerate(training_iterator):
            if seen_classes is None:
                seen_classes = labels.unique()
            else:
                seen_classes = torch.cat([seen_classes, labels], dim=0).unique()
            data_input = data.to(device)
            labels_input = labels.to(device)
            loss, logits, sample_loss = learner.training_loss(data=data_input, labels=labels_input,
                                                              **learner_kwargs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_step += 1
            with torch.no_grad():
                if (log_eval_every > 0 and current_step % log_eval_every == 0)\
                        or (current_step in log_eval_steps):
                    results_dict = log_step(current_step, seen_classes=seen_classes)
                    if all_results is None:
                        all_results = {k: [v] for k, v in results_dict.items()}

                    else:
                        for k, v in results_dict.items():
                            all_results[k].append(v)
                    learner.train()
                    last_eval_step = current_step
    if not (last_eval_step == current_step):
        with torch.no_grad():
            if log_eval_every > 0:

                results_dict = log_step(current_step, seen_classes=seen_classes)
            else:
                results_dict = evaluate_iterators(learner=learner,
                                                  evaluation_iterators=evaluation_iterators,
                                                  learner_kwargs=learner_kwargs, device=device)
                num_seen = seen_classes.size()[0]
                results_dict['seen_classes'] = num_seen
            if all_results is None:
                all_results = {k: [v] for k, v in results_dict.items()}
            else:
                for k, v in results_dict.items():
                    all_results[k].append(v)
    return learner, all_results


def load_config(experiment_path):
    hydra_folder = osp.join(experiment_path, ".hydra")
    with initialize(config_path=hydra_folder):
        cfg = compose(config_name="config")
    return cfg


def load_config_hydra_safe(experiment_path):
    hydra_folder = osp.join(experiment_path, ".hydra")
    cfg = OmegaConf.load(osp.join(hydra_folder, 'config.yaml'))
    return cfg


def load_learner_cfg(experiment_path, model_name=None, iteration=None,
                     hydra_safe=False, device=None):
    if hydra_safe:
        cfg = load_config_hydra_safe(experiment_path=experiment_path)
    else:
        cfg = load_config(experiment_path=experiment_path)
    learner = hydra.utils.instantiate(cfg.learner)
    if model_name is None:
        if iteration is not None:
            model_name = "learner_{}.net".format(iteration)
        else:
            files = [f for f in os.listdir('.') if f[-4:] == '.net']
            print(files)
            print('WARNING: loading randomly initialized learner')
            return learner, cfg
    model_path = osp.join(experiment_path, model_name)
    if device is not None:
        learner = torch.load(model_path, map_location=device)
    else:
        learner = torch.load(model_path)
    return learner, cfg


def evaluate_continual_learner(learner,
                               optimizer,
                               dataset_name,
                               total_tasks,
                               classes_per_task,
                               training_batch_size,
                               evaluation_batch_size,
                               device,
                               freeze_components,
                               reinit_components,
                               reinit_fn=original_init,
                               reinit_memory=True,
                               epochs=1,
                               learner_kwargs={},
                               log_eval_every=-1,
                               logging_subdir='data',
                               meta_test_on_train=False):
    work_dir = Path.cwd()
    print('workspace: {}'.format(work_dir))
    start_evaluation_time = time.time()
    evaluation_logger, *_ = setup_loggers(d=logging_subdir, namespaces=['evaluation'],
                                      colors=['white'], )

    dataset_train = df.DatasetFactory.get_dataset(
        dataset_name, background=meta_test_on_train, train=True, all=False)
    dataset_test = df.DatasetFactory.get_dataset(
        dataset_name, background=meta_test_on_train, train=False, all=False)

    training_iterator, evaluation_train_iterator, evaluation_test_iterator = (
        prepare_dataset_iterators(dataset_name=dataset_name,
                                  dataset_train=dataset_train,
                                  dataset_test=dataset_test,
                                  total_tasks=total_tasks,
                                  classes_per_task=classes_per_task,
                                  training_batch_size=training_batch_size,
                                  evaluation_batch_size=evaluation_batch_size, ))

    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    learner.to(device=device)
    freeze_model_components(model=learner, components=freeze_components,
                            verbose=True)
    reinit_model_components(model=learner, components=reinit_components,
                            reinit_fn=reinit_fn, reinit_memory=reinit_memory,
                            verbose=True)

    learner, all_results = learn_continual_classifier(
        learner=learner,
        optimizer=optimizer,
        training_iterator=training_iterator,
        evaluation_iterators={'train_set': evaluation_train_iterator,
                              'test_set': evaluation_test_iterator, },
        evaluation_logger=evaluation_logger,
        epochs=epochs,
        device=device,
        learner_kwargs=learner_kwargs,
        log_eval_every=log_eval_every)
    print('Total evaluation time: {}'.format(time.time() - start_evaluation_time))
    return learner, all_results



def evaluate_legacy_continual_learner(
                               learner,
                               optimizer,
                               dataset_name,
                               total_tasks,
                               classes_per_task,
                               training_batch_size,
                               evaluation_batch_size,
                               device,
                               total_frozen_layers=13,
                               epochs=1,
                               learner_kwargs={},
                               log_eval_every=-1, ):
    work_dir = Path.cwd()
    print('workspace: {}'.format(work_dir))
    start_evaluation_time = time.time()
    d = 'temp/eval/{}'.format(time.time())
    evaluation_logger, *_ = setup_loggers(d=d, namespaces=['evaluation'],
                                      colors=['white'], )

    dataset_train = df.DatasetFactory.get_dataset(
        dataset_name, background=False, train=True, all=False)
    dataset_test = df.DatasetFactory.get_dataset(
        dataset_name, background=False, train=False, all=False)

    training_iterator, evaluation_train_iterator, evaluation_test_iterator = (
        prepare_dataset_iterators(dataset_name=dataset_name,
                                  dataset_train=dataset_train,
                                  dataset_test=dataset_test,
                                  total_tasks=total_tasks,
                                  classes_per_task=classes_per_task,
                                  training_batch_size=training_batch_size,
                                  evaluation_batch_size=evaluation_batch_size, ))

    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    learner.to(device=device)
    freeze_legacy_learner(model=learner, total_frozen_layers=total_frozen_layers)

    learner, all_results = learn_continual_classifier(
        learner=learner,
        optimizer=optimizer,
        training_iterator=training_iterator,
        evaluation_iterators={'train_set': evaluation_train_iterator,
                              'test_set': evaluation_test_iterator, },
        evaluation_logger=evaluation_logger,
        epochs=epochs,
        device=device,
        learner_kwargs=learner_kwargs,
        log_eval_every=log_eval_every)
    print('Total evaluation time: {}'.format(time.time() - start_evaluation_time))
    return learner, all_results
