import copy
import os.path

import hydra
from omegaconf import OmegaConf

from training_utils import set_seed
from evaluation_function import load_learner_cfg

@hydra.main(config_path='cfgs/eval', config_name='config_tune_lr')
def main(cfg):
    tune_lr(cfg)
def tune_lr(cfg):
    set_seed(cfg.seed)
    total_subfolders = len(cfg.relative_subdir.split('/'))
    relative_experiment_path = '../'*total_subfolders # go down as many folders as appended
    learner, learner_cfg = load_learner_cfg(experiment_path=relative_experiment_path,
                                            model_name=cfg.model_name,
                                            iteration=cfg.iteration,
                                            hydra_safe=True,
                                            device=cfg.device,)
    eval_kwargs = cfg.eval_kwargs
    if eval_kwargs is None:
        eval_kwargs = learner_cfg.eval_kwargs
    else:
        eval_kwargs = OmegaConf.merge(learner_cfg.eval_kwargs, eval_kwargs)
        print(eval_kwargs)

    print(eval_kwargs['freeze_components'])


    optimizer_cfgs = copy.deepcopy(eval_kwargs['optimizer'])
    del eval_kwargs['optimizer']

    if cfg.tune_start_lr:
        current_lr = cfg.tune_start_lr
    else:
        current_lr = optimizer_cfgs.lr
    # make unique
    lr_choices = list(set(cfg.tune_base_options + cfg.tune_additional_options))
    lr_dict = {}
    if current_lr not in lr_choices:
        lr_choices.append(current_lr)
    lr_choices.sort()
    num_choices = len(lr_choices)

    best_lr = None
    best_accuracy = -1

    current_index = None
    for i in range(num_choices):
        if lr_choices[i] == current_lr:
            current_index = i

    def get_lr_results(current_lr):
        optimizer_cfgs['lr'] = current_lr
        optimizer = hydra.utils.instantiate(optimizer_cfgs, params=learner.parameters())
        print(optimizer)
        logging_subdir = os.path.join('learning_rate_{}'.format(current_lr), cfg.timestamp)
        trained_learner, all_results = hydra.utils.call(
            cfg.evaluation_function, learner=learner, optimizer=optimizer,
            logging_subdir=logging_subdir, **eval_kwargs)
        if cfg.tune_criterion == 'final':
            return all_results['test_set_accuracy'][-1], all_results
        else:
            raise NotImplementedError

    to_check_idx = current_index
    all_lr_results_dict = {}
    for i in range(cfg.tune_max_iter):
        current_lr = lr_choices[current_index]
        lr_dict[current_lr], all_lr_results = get_lr_results(current_lr)
        all_lr_results_dict[current_lr] = all_lr_results
        if lr_dict[current_lr] > best_accuracy:
            best_accuracy = lr_dict[current_lr]
            best_lr = current_lr
        if current_index < (len(lr_choices)-1):
            larger_lr = lr_choices[current_index + 1]
            if larger_lr in lr_dict:
                if lr_dict[larger_lr] > lr_dict[current_lr]:
                    lr_choices = lr_choices[current_index+1:]
                    current_index = 0
                    to_check_idx = 2
                else:
                    lr_choices = lr_choices[:current_index]
        if current_index != 0:
            smaller_lr = lr_choices[current_index-1]
            if smaller_lr in lr_dict:
                if lr_dict[smaller_lr] < lr_dict[current_lr]:
                    lr_choices = lr_choices[current_index:]
                    current_index = 0
                    to_check_idx = 1
                else:
                    lr_choices = lr_choices[:current_index]
                    to_check_idx = current_index - 1
            else:
                    to_check_idx = current_index - 1
        if (to_check_idx >= 0) and (to_check_idx < len(lr_choices)):
            if lr_choices[to_check_idx] not in lr_dict:
                print('remaining choices {}'.format(lr_choices))
                current_index = to_check_idx
            else:
                break
        else:
            break
    print('Best lr {} --- Performance {}'.format(best_lr, best_accuracy))
    return best_accuracy, best_lr, all_lr_results

if __name__ == '__main__':
    main()
