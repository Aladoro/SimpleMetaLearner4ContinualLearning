import time
import torch
import numpy as np
import datasets.datasetfactory as df
from datasets.task_sampler import get_sampler

from pathlib import Path

from logging_utils import Metrics, setup_loggers
from training_utils import iterator_trainer_statistics, to_numpy


def log_dict(logger, dictionary, prefix=''):
    for k, v in dictionary.items():
        logger.log_key_val(prefix + k, v)


def continual_meta_training(dataset_name,
                            trainer,
                            training_steps,
                            classes_per_task=1,
                            task_batch_size=20,
                            all_tasks_batch_size=64,
                            meta_batch_size=1,
                            separate_validation_data=False,
                            all_data_meta=False,
                            task_validation_batch_size=5,
                            detect_anomaly=False,
                            device=None,
                            evaluation_batch_size=128,
                            log_training_every=40,
                            log_evaluation_every=2000,
                            save_weights_every=10000):
    if detect_anomaly:
        print('WARNING: running with activated anomaly detection')
        torch.autograd.set_detect_anomaly(True)
    work_dir = Path.cwd()
    print('workspace: {}'.format(work_dir))

    train_logger, test_logger = setup_loggers(d='data',
                                              namespaces=['train', 'test'],
                                              colors=['white', 'yellow'],)

    dataset_train = df.DatasetFactory.get_dataset(
        dataset_name, background=True, train=True,
        all=not separate_validation_data)
    if all_data_meta:
        assert separate_validation_data
    if all_data_meta:
        dataset_test = df.DatasetFactory.get_dataset(
            dataset_name, background=True, train=False,
            all=True)
    else:
        dataset_test = df.DatasetFactory.get_dataset(
            dataset_name, background=True, train=False,
            all=not separate_validation_data)

    classes_idxs = list(range(np.max(dataset_train.targets)))

    # Iterators used for evaluation
    iterator_test = torch.utils.data.DataLoader(dataset_test, batch_size=evaluation_batch_size,
                                                shuffle=True, num_workers=1)

    iterator_train = torch.utils.data.DataLoader(dataset_train, batch_size=evaluation_batch_size,
                                                 shuffle=True, num_workers=1)


    sampler = get_sampler(dataset_name, classes_idxs, trainset=dataset_train,
                          testset=dataset_test, batch_size=task_batch_size,
                          validation_batch_size=task_validation_batch_size,
                          all_tasks_batch_size=all_tasks_batch_size)

    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    trainer.set_device(device=device)

    training_metrics = Metrics('training_accuracy', 'meta_loss',
                               'training_time')

    start_training_time = time.time()
    test_logger.log_key_val('step', 0)
    step_start_eval_time = time.time()
    training_data_stats = iterator_trainer_statistics(trainer=trainer,
                                                      iterator=iterator_train)
    test_data_stats = iterator_trainer_statistics(trainer=trainer,
                                                  iterator=iterator_test)
    log_dict(test_logger, training_data_stats, prefix='train_data_')
    log_dict(test_logger, test_data_stats, prefix='test_data_')

    step_end_eval_time = time.time()
    evaluation_time = step_end_eval_time - step_start_eval_time
    test_logger.log_key_val('evaluation_time', evaluation_time)
    test_logger.log_iteration()
    for step in range(training_steps):
        step_start_training_time = time.time()
        task_iterators = []
        if separate_validation_data:
            validation_task_iterators = []
        else:
            validation_task_iterators = None
        all_tasks_iterators = []
        for i in range(meta_batch_size):
            sampled_classes = np.random.choice(classes_idxs,
                                               size=classes_per_task,
                                               replace=False)
            for c in sampled_classes:
                task_iterators.append(sampler.sample_task([c], train=True))
                if separate_validation_data:
                    validation_task_iterators.append(
                        sampler.sample_task([c], train=False))

            all_tasks_iterator = sampler.sample_all_tasks()
            all_tasks_iterators.append(all_tasks_iterator)

        training_accuracy, meta_loss = trainer.meta_train(
            task_iterators=task_iterators,
            validation_task_iterators=validation_task_iterators, # todo
            all_tasks_iterators=all_tasks_iterators)
        if trainer.meta_lr_scheduler is not None:
            trainer.meta_lr_scheduler.step()
        step_end_training_time = time.time()

        training_time = step_end_training_time - step_start_training_time
        training_metrics.update(training_accuracy=training_accuracy,
                                meta_loss=meta_loss, training_time=np.around(training_time, 3))

        if (step + 1) % log_training_every == 0:
            train_logger.log_key_val('step', step + 1)
            current_training_metrics = training_metrics.get()
            training_metrics.reset()
            log_dict(logger=train_logger, dictionary=current_training_metrics)
            log_dict(logger=train_logger, dictionary=trainer.logging_stats())
            train_logger.log_iteration()
        if (step + 1) % log_evaluation_every == 0 or step == training_steps:
            test_logger.log_key_val('step', step + 1)
            step_start_eval_time = time.time()
            training_data_stats = iterator_trainer_statistics(trainer=trainer,
                                                              iterator=iterator_train)
            test_data_stats = iterator_trainer_statistics(trainer=trainer,
                                                          iterator=iterator_test)
            log_dict(test_logger, training_data_stats, prefix='train_data_')
            log_dict(test_logger, test_data_stats, prefix='test_data_')

            step_end_eval_time = time.time()
            evaluation_time = step_end_eval_time - step_start_eval_time
            test_logger.log_key_val('evaluation_time', evaluation_time)
            test_logger.log_iteration()
        if (step + 1) % save_weights_every == 0 or step == training_steps:
            print('saving model for iteration {}'.format(step+1))
            torch.save(trainer.learner, 'learner_{}.net'.format(step + 1))

    print('Total execution time: {}'.format(time.time() - start_training_time))
