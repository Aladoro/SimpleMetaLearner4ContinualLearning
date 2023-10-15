import os
import sys

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from hydra import compose, initialize
from training_utils import set_seed
from tune_lr import tune_lr

from eval_all import main as eval_all_fn

EVAL_DICT = {}
EVAL_ALL = False
TUNE_LR = False
@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    global EVAL_DICT
    global EVAL_ALL
    global TUNE_LR
    if cfg.tune_lr:
        EVAL_DICT['experiment_path'] = os.getcwd()
        EVAL_DICT['iteration'] = cfg.training_steps
        TUNE_LR = True
    if cfg.eval_all:
        EVAL_DICT['iteration'] = cfg.training_steps
        EVAL_DICT['experiment_path'] = os.getcwd()
        EVAL_ALL = True
    set_seed(cfg.seed)
    trainer = hydra.utils.instantiate(cfg.trainer)
    hydra.utils.call(cfg.training_function, trainer=trainer)
if __name__ == '__main__':
    main()
    if TUNE_LR:
        with initialize(config_path="cfgs/eval", job_name="tune_lr"):
            overrides_list = ["{}={}".format(k, v) for k, v in EVAL_DICT.items()]
            cfg_eval = compose(config_name="config_tune_lr", overrides=overrides_list)
            eval_base_dir = os.path.join(cfg_eval.experiment_path, cfg_eval.relative_subdir)
            os.makedirs(eval_base_dir, exist_ok=True)
            base_dir = os.getcwd()
            os.chdir(eval_base_dir)
            best_accuracy, best_lr, all_lr_results = tune_lr(cfg_eval)
            os.chdir(base_dir)
    if EVAL_ALL:
        EVAL_DICT['experiment_path'] = os.path.dirname(EVAL_DICT['experiment_path'])
        if 'iteration' in EVAL_DICT:
            del EVAL_DICT['iteration']
        with initialize(config_path="cfgs/eval", job_name="eval_all"):
            overrides_list = ["{}={}".format(k, v) for k, v in EVAL_DICT.items()]
            cfg_eval = compose(config_name="config_eval_all", overrides=overrides_list)
            eval_base_dir = os.path.join(cfg_eval.experiment_path, cfg_eval.relative_subdir)
            os.makedirs(eval_base_dir, exist_ok=True)
            base_dir = os.getcwd()
            os.chdir(eval_base_dir)
            eval_all_fn(cfg_eval)
            os.chdir(base_dir)
