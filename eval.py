import os

import hydra
from omegaconf import OmegaConf

from training_utils import set_seed
from evaluation_function import load_learner_cfg


@hydra.main(config_path='cfgs/eval', config_name='config_eval')
def main(cfg):
    set_seed(cfg.seed)
    total_subfolders = len(cfg.relative_subdir.split('/'))
    relative_experiment_path = '../'*total_subfolders
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


    optimizer = hydra.utils.instantiate(eval_kwargs.optimizer,
                                        params=learner.parameters())
    lr = eval_kwargs.optimizer.lr
    del eval_kwargs['optimizer']

    logging_subdir = os.path.join('learning_rate_{}'.format(lr), cfg.timestamp)
    hydra.utils.call(
        cfg.evaluation_function, learner=learner, optimizer=optimizer,
        logging_subdir=logging_subdir, **eval_kwargs)


if __name__ == '__main__':
    main()
