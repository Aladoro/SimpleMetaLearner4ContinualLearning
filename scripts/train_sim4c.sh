#!/bin/bash

RANDOM=$$

R=$(($RANDOM%1000));
echo "Seed $R";
python train.py eval_all=true seed=$R learner=fixed_rln meta_trainer=meta_classifier_collect_fx_ts meta_loss=maml save_weights_every=5000 training_steps=100000 meta_reset_random_classes=5 logging_folder=exp_local/ algorithm_name=sim4c  inner_update_steps=1;

