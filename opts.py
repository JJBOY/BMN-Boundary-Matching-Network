import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--mode',
        type=str,
        default='train')
    parser.add_argument(
        '--module',
        type=str,
        default='BMN')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoint')
    parser.add_argument(
        '--training_lr',
        type=float,
        default=0.001)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=5e-5)

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
        default=10)
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.1)
    # Overall Dataset settings
    parser.add_argument(
        '--video_info',
        type=str,
        default="/home/zenghao/BSN/data/activitynet_annotations/video_info_new.csv")
    parser.add_argument(
        '--video_anno',
        type=str,
        default="/home/zenghao/BSN/data/activitynet_annotations/anet_anno_action.json")
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=100)
    parser.add_argument(
        '--boundary_ratio',
        type=float,
        default=0.1)
    parser.add_argument(
        '--feature_path',
        type=str,
        default="/home/zenghao/BSN/data/activitynet_feature_cuhk/")

    # Base model settings
    parser.add_argument(
        '--base_feat_dim',
        type=int,
        default=400)
    parser.add_argument(
        '--base_out_dim',
        type=int,
        default=128)

    # TEM model settings
    parser.add_argument(
        '--tem_feat_dim',
        type=int,
        default=128)
    parser.add_argument(
        '--tem_hidden_dim',
        type=int,
        default=256)

    # PEM model settings
    parser.add_argument(
        '--pem_feat_dim',
        type=int,
        default=128)
    parser.add_argument(
        '--pem_hidden_dim',
        type=int,
        default=512)
    parser.add_argument(
        '--pem_duration',
        type=int,
        default=100)
    parser.add_argument(
        '--pgm_threshold',
        type=float,
        default=0.5)
    
    # TEM Training settings
    parser.add_argument(
        '--tem_match_thres',
        type=float,
        default=0.5)

    # PEM Training settings
    parser.add_argument(
        '--bm_mask_path',
        type=str,
        default="./BM_mask.npy")
    parser.add_argument(
        '--pem_u_ratio_m',
        type=float,
        default=2)
    parser.add_argument(
        '--pem_u_ratio_l',
        type=float,
        default=1)
    parser.add_argument(
        '--pem_high_iou_thres',
        type=float,
        default=0.6)
    parser.add_argument(
        '--pem_low_iou_thres',
        type=float,
        default=0.2)
    parser.add_argument(
        '--pem_sample_num',
        type=int,
        default=32)

    # Post processing
    parser.add_argument(
        '--post_process_top_K',
        type=int,
        default=100)
    parser.add_argument(
        '--post_process_thread',
        type=int,
        default=8)
    parser.add_argument(
        '--soft_nms_alpha',
        type=float,
        default=0.75)
    parser.add_argument(
        '--soft_nms_low_thres',
        type=float,
        default=0.65)
    parser.add_argument(
        '--soft_nms_high_thres',
        type=float,
        default=0.9)
    parser.add_argument(
        '--result_file',
        type=str,
        default="./output/result_proposal.json")
    parser.add_argument(
        '--save_fig_path',
        type=str,
        default="./output/evaluation_result.jpg")

    args = parser.parse_args()

    return args
