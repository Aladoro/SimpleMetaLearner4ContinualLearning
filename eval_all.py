import os
from collections import OrderedDict

import hydra
import numpy as np
from omegaconf import OmegaConf
from tabulate import tabulate

from training_utils import set_seed
from logging_utils import save_dict_of_lists
from evaluation_function import load_learner_cfg


@hydra.main(config_path='cfgs/eval', config_name='config_eval_all')
def main(cfg):
    set_seed(cfg.seed)

    total_subfolders = len(cfg.relative_subdir.split('/'))
    relative_experiment_path = '../'*total_subfolders

    directories = os.listdir(relative_experiment_path)
    all_results_exps = OrderedDict()
    all_results_metric = dict()
    all_results_iter = dict()
    iterations_list = None
    for dir in directories:
        dir_relative_experiment_path = os.path.join(relative_experiment_path, dir)
        if os.path.isdir(dir_relative_experiment_path):
            model_subdirs = [subd for subd in os.listdir(dir_relative_experiment_path) if subd.endswith('.net')]
            model_iters = []
            for subd in model_subdirs:
                split_name = (os.path.splitext(subd)[0]).split('_')
                assert len(split_name) == 2
                iteration = int(split_name[1])
                model_iters.append(iteration)
            latest_iter = 0
            results_per_iter = None
            if cfg.scratch_mode:
                model_iters = [None]
            for iteration in sorted(model_iters):
                exp_name = os.path.basename(os.path.normpath(dir_relative_experiment_path))
                learner, learner_cfg = load_learner_cfg(experiment_path=dir_relative_experiment_path,
                                                        iteration=iteration,
                                                        hydra_safe=True,
                                                        device=cfg.device,)
                if (cfg.min_iteration_steps_diff is not None) and (not cfg.scratch_mode):
                    if iteration < (latest_iter + cfg.min_iteration_steps_diff):
                        continue
                    latest_iter = iteration
                eval_kwargs = cfg.eval_kwargs
                if eval_kwargs is None:
                    eval_kwargs = learner_cfg.eval_kwargs
                else:
                    eval_kwargs = OmegaConf.merge(learner_cfg.eval_kwargs, eval_kwargs)  # Priority to eval_kwargs
                if cfg.scratch_mode:
                    eval_kwargs.freeze_components = []
                    eval_kwargs.reinit_components = ['rln_layers', 'pln_layers']

                optimizer = hydra.utils.instantiate(eval_kwargs.optimizer,
                                                    params=learner.parameters())
                lr = eval_kwargs.optimizer.lr
                del eval_kwargs['optimizer']
                all_eval_seed_test = []
                all_eval_seed_train = []
                all_eval_seed_lists = None
                if len(cfg.eval_seeds) == 0:
                    cfg.eval_seeds = [np.random.randint(1000)]
                base_loc = os.path.join(cfg.timestamp, 'learning_rate_{}'.format(lr), dir)
                for seed in cfg.eval_seeds:
                    set_seed(seed)

                    logging_subdir = os.path.join(base_loc, '{}_{}'.format(iteration, seed))
                    if cfg.scratch_mode:
                        logging_subdir = os.path.join(base_loc, '{}_{}'.format('scratch', seed))
                    learner, all_results = hydra.utils.call(
                        cfg.evaluation_function, learner=learner, optimizer=optimizer,
                        logging_subdir=logging_subdir, **eval_kwargs)
                    test_set_accuracies = np.array(all_results['test_set_accuracy'])
                    train_set_accuracies = np.array(all_results['train_set_accuracy'])
                    all_eval_seed_test.append(test_set_accuracies)
                    all_eval_seed_train.append(train_set_accuracies)
                    if all_eval_seed_lists:
                        for k, v in all_results.items():
                            all_eval_seed_lists[k].append(v)
                    else:
                        all_eval_seed_lists = {k: [v] for k, v in all_results.items()}
                all_eval_seed_res = {k: np.mean(v, axis=0) for k, v in all_eval_seed_lists.items()}
                print(all_eval_seed_res)
                print(all_eval_seed_res.keys())
                print('Experiment {} -- ITERATION {}'.format(exp_name, iteration))
                logged_metrics = ['seen_classes', 'train_set_accuracy', 'test_set_accuracy']
                tabulate_rows = [[lm]+list(all_eval_seed_res[lm]) for lm in logged_metrics]
                if iterations_list is None:
                    iterations_list = tabulate_rows[0]
                print(tabulate(tabulate_rows, headers="firstrow"))
                if results_per_iter is None:
                    results_per_iter = OrderedDict(seen_classes=all_eval_seed_res['seen_classes'])
                results_per_iter[iteration] = all_eval_seed_res['test_set_accuracy']
            if results_per_iter is not None:
                best = 0.0
                best_iter = None
                best_results = None
                for i, (iteration, results) in enumerate(results_per_iter.items()):
                    if i > 0:
                        if cfg.metric == 'final':
                            iter_results = results[-1]
                        else:
                            raise NotImplementedError
                        if iter_results > best:
                            best = iter_results
                            best_iter = iteration
                            best_results = results
                all_results_exps[dir] = best_results
                all_results_metric[dir] = best
                all_results_iter[dir] = best_iter
            else:
                continue
            save_dict_of_lists(filename=os.path.join(cfg.timestamp, '{}_summary'.format(dir)), dictionary=results_per_iter)
            print('-------------------------------------------------------------------------')
            print('Experiment {} conclusion'.format(dir))  # TODO add format
            print(results_per_iter)
            tabulate_rows = [[k] + list(v) for k, v in results_per_iter.items()]
            print(tabulate(tabulate_rows, headers="firstrow"))
    assert iterations_list is not None, 'no experiments folder found'
    print('-------------------------------------------------------------------------')
    print('All seeds results')
    print(all_results_exps)
    all_results_exps['MEAN'] = np.mean(list(all_results_exps.values()), axis=0)
    all_results_iter['MEAN'] = np.mean(list(all_results_iter.values()), axis=0)
    all_results_metric['MEAN'] = np.mean(list(all_results_metric.values()), axis=0)
    headers = iterations_list + ['Best ITER', 'METRIC score']
    tabulate_rows = [[k] + list(v) + [all_results_iter[k], all_results_metric[k]] for k, v in all_results_exps.items()]
    all_results_exps[iterations_list[0]] = iterations_list[1:]
    save_dict_of_lists(filename=os.path.join(cfg.timestamp, 'seeds_results'), dictionary=all_results_exps)
    print(tabulate(tabulate_rows, headers=headers))


if __name__ == '__main__':
    main()
