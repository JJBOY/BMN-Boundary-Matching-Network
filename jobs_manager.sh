#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1 python main.py --mode train --experiment_name og_dropout
# CUDA_VISIBLE_DEVICES=0,1 python main.py --mode train --experiment_name og_reverse --reverse 1
CUDA_VISIBLE_DEVICES=0,1 python main.py --mode train --experiment_name tsm_reverse --reverse 1 --max_shift 5 --shift_prob 0.2
CUDA_VISIBLE_DEVICES=0,1 python main.py --mode train --experiment_name se --s_and_e True
CUDA_VISIBLE_DEVICES=0,1 python main.py --mode train --experiment_name se_reverse --reverse 1 --s_and_e True
CUDA_VISIBLE_DEVICES=0,1 python main.py --mode train --experiment_name tsm_se --max_shift 5 --shift_prob 0.2 --s_and_e True
CUDA_VISIBLE_DEVICES=0,1 python main.py --mode train --experiment_name tsm_se_reverse --reverse 1 --max_shift 5 --shift_prob 0.2 --s_and_e True
CUDA_VISIBLE_DEVICES=0,1 python main.py --mode train --experiment_name tsm --max_shift 5 --shift_prob 0.2

'''
CUDA_VISIBLE_DEVICES=0,1 python main.py --mode inference --forward_model og_dropout
CUDA_VISIBLE_DEVICES=0,1 python main.py --mode inference --forward_model og_dropout --reverse_model og_reverse --ensemble 1
'''