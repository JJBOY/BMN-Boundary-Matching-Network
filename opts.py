import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--mode',
        type=str,
        default='train')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoint')
    parser.add_argument(
        '--training_lr',
        type=float,
        default=0.001)  # 0.001
    parser.add_argument(
        '--optimizer',
        type=str,
        default="Adam")  # 0.001
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.4)

    parser.add_argument(
        '--train_epochs',
        type=int,
        default=15)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16)
    parser.add_argument(
        '--step_size',
        type=int,
        default=7)  # 7
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.5)  # 0.1
    parser.add_argument(
        '--shift_prob',
        type=float,
        default=0)
    parser.add_argument(
        '--max_shift',
        type=int,
        default=0) 
    parser.add_argument(
        '--patience',
        type=int,
        default=2)  

    # Random seed for reproducibility
    parser.add_argument(
        '--random_seed',
        type=int,
        default=1)

    # Overall Dataset settings
    parser.add_argument(
        '--video_info',
        type=str,
        default="data/activitynet_annotations/activity_net_1_3_new.json")
    parser.add_argument(
        '--video_anno',
        type=str,
        default="data/activitynet_annotations/activitynet_13_annotations.json")
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=100)
    parser.add_argument(
        '--feature_path',
        type=str,
        default="data/activitynet_feature_cuhk/fix_feat_100/")

    parser.add_argument(
        '--num_sample',
        type=int,
        default=32)
    parser.add_argument(
        '--num_sample_perbin',
        type=int,
        default=3)
    parser.add_argument(
        '--prop_boundary_ratio',
        type=int,
        default=0.5)

    parser.add_argument(
        '--feat_dim',
        type=int,
        default=400)

    # Post processing
    parser.add_argument(
        '--post_process_thread',
        type=int,
        default=8)
    parser.add_argument(
        '--soft_nms_alpha',
        type=float,
        default=0.4)
    parser.add_argument(
        '--soft_nms_low_thres',
        type=float,
        default=0.5)
    parser.add_argument(
        '--soft_nms_high_thres',
        type=float,
        default=0.9)
    parser.add_argument(
        '--result_file',
        type=str,
        default="./output/result_proposal_new.json")
    parser.add_argument(
        '--save_fig_path',
        type=str,
        default="./output/evaluation_result_new.jpg")
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='debug'
    )
    parser.add_argument(
        '--forward_model',
        type=str,
        default='og_run'
        # default='og_run'
    )
    parser.add_argument(
        '--reverse_model',
        type=str,
        # default='reverse_OG_fixed_bug_w_dropout_noFlipConfMap'
        default='og_reversed'
    )
    parser.add_argument(
        '--reverse',
        type=int,
        default=0
    )
    parser.add_argument(
        '--ensemble',
        type=int,
        default=0
    )
    parser.add_argument(
        '--s_and_e',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--se_hidden_dim',
        type=int,
        default=100
    )

    args = parser.parse_args()

    return args

