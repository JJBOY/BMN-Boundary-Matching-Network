# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import torch.utils.data as data
import torch
from scipy.spatial import distance
from utils import ioa_with_anchors, iou_with_anchors


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


class VideoDataSet(data.Dataset):
    def __init__(self, opt, subset="train", reverse=False):
        self.temporal_scale = opt["temporal_scale"]  # 100
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = opt["mode"]
        self.feature_path = opt["feature_path"]
        self.video_anno_path = opt["video_anno"]
        self.video_info_path = opt["video_info"]
        self.shift_prob = opt["shift_prob"]
        self.max_shift = opt["max_shift"]
        self._getDatasetDict()
        self.anchor_xmin = [self.temporal_gap * (i - 0.5) for i in range(self.temporal_scale)]
        self.anchor_xmax = [self.temporal_gap * (i + 0.5) for i in range(self.temporal_scale)]
        self.reverse = reverse

    def _getDatasetDict(self):
        # anno_df = pd.read_csv(self.video_info_path)
        # anno_database = load_json(self.video_anno_path)
        # self.video_dict = {}
        # for i in range(len(anno_df)):
        #     video_name = anno_df.video.values[i]
        #     video_info = anno_database[video_name]
        #     video_subset = anno_df.subset.values[i]
        #     if self.subset in video_subset:
        #         self.video_dict[video_name] = video_info
        # self.video_list = list(self.video_dict.keys())
        # print("%s subset video numbers: %d" % (self.subset, len(self.video_list)))

        anno_database1 = load_json(self.video_anno_path)
        anno_database2 = load_json(self.video_info_path)
        anno_database2 = anno_database2['database']

        self.video_dict = {}

        # Not sure why they don't save testing labels so here they are
        self.rest_dict = {}

        # for i in range(len(anno_df)):
        for key, items in anno_database1.items():
            video_name = key

            video_info = items
            temp_dict = anno_database2[key[2:]]
            video_info['resolution'] = temp_dict['resolution']
            video_info['url'] = temp_dict['url']

            video_subset = items['subset']
            # video_name = anno_df.video.values[i]
            # video_info = anno_database[video_name]
            # video_subset = anno_df.subset.values[i]
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
            else:
                self.rest_dict[video_name] = video_info

        self.video_list = list(self.video_dict.keys())
        print("%s subset video numbers: %d" % (self.subset, len(self.video_list)))

    def __getitem__(self, index):
        video_data = self._load_file(index)
        if self.mode == "train":
            match_score_start, match_score_end, confidence_score = self._get_train_label(index, self.anchor_xmin,
                                                                                         self.anchor_xmax)
            return video_data, confidence_score, match_score_start, match_score_end
        else:
            return index, video_data

    def _add_global_features(self, video_data):
            global_mean = np.mean(video_data, axis=0)
            global_mean_repeated = np.tile(global_mean,(100,1))
            video_data_with_global = np.concatenate((video_data, global_mean_repeated), axis=1)
            return video_data_with_global

    def _get_shifted_features(self, feats, max_shift=10, shift_prob=0.5):
        if max_shift == 0:
            return feats
        num_timesteps, num_feats = feats.shape

        shifted_feats = np.zeros_like(feats)
        features_to_shift = (np.random.uniform(size=num_feats) < shift_prob) * 1
        shift_left_or_right = ((np.random.uniform(size=num_feats) < 0.5)*-2) + 1 # equal prob of shifting left/right

        num_shifts = np.random.randint(low=1, high=max_shift + 1, size=num_feats)
        num_shifts = num_shifts * shift_left_or_right * features_to_shift

        for f in range(num_feats):
            num_shift = num_shifts[f]
            if num_shift > 0:
                shifted_feats[num_shift:, f] = feats[:-num_shift, f] # positive shift -> shift right
            elif num_shift < 0:
                shifted_feats[0:num_shift, f] = feats[-num_shift:, f] # shift left
            else:
                shifted_feats[:, f] = feats[:, f] # just copy
        return shifted_feats

    def _compute_similarity(self, feats, sim_type = "cosine"): 
        print(feats.shape)
        similarity_scores = np.zeros((feats.shape[0], 1))
        for i in range(1, feats.shape[0]):
            similarity_scores[i] = 1 - distance.cosine(feats[i, :], feats[i-1, :])
        feats_with_sim = np.concatenate((feats, similarity_scores), axis=1)
        print(feats_with_sim.shape)
        raise "bye for now"
        return feats_with_sim

    def _load_file(self, index):
        video_name = self.video_list[index]
        # video_df = pd.read_csv(self.feature_path + "csv_mean_" + str(self.temporal_scale) + "/" + video_name + ".csv")
        # video_data = video_df.values[:, :]
        video_data = np.load(self.feature_path + video_name + ".npy")
        # print(f'video_data: {video_data.shape}')
        # print(f'test: {np.mean(video_data, axis=0).shape}')

        '''
        Reverse frame order
        '''
        if self.reverse != 0:
            video_data = video_data[::-1].copy()

        if self.subset == "validation":
            feats = video_data
        elif self.subset == "train": 
            feats = self._get_shifted_features(video_data, shift_prob=self.shift_prob, max_shift=self.max_shift)
        # feats = video_data
        # feats = self._compute_similarity(feats)
        # feats = self._add_global_features(feats)

        feats = torch.Tensor(feats)
        feats = torch.transpose(feats, 0, 1).float()
        return feats.float()

    def _get_train_label(self, index, anchor_xmin, anchor_xmax):
        video_name = self.video_list[index]
        video_info = self.video_dict[video_name]
        video_frame = video_info['duration_frame']
        video_second = video_info['duration_second']
        feature_frame = video_info['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second  # there are some frames not used
        video_labels = video_info['annotations']  # the measurement is second, not frame

        ##############################################################################################
        # change the measurement from second to percentage
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)

            '''
            Flip start and end
            '''
            if self.reverse != 0:
                gt_bbox.append([1 - tmp_end, 1 - tmp_start])
            else:
                gt_bbox.append([tmp_start, tmp_end])

            # gt_bbox.append([tmp_start, tmp_end])

        ####################################################################################################
        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.temporal_gap  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        #####################################################################################################

        gt_iou_map = np.zeros([self.temporal_scale, self.temporal_scale])
        for i in range(self.temporal_scale):
            for j in range(i, self.temporal_scale):
                gt_iou_map[i, j] = np.max(
                    iou_with_anchors(i * self.temporal_gap, (j + 1) * self.temporal_gap, gt_xmins, gt_xmaxs))
        gt_iou_map = torch.Tensor(gt_iou_map)

        ##########################################################################################################
        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        ############################################################################################################
        return match_score_start, match_score_end, gt_iou_map

    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':
    import opts

    opt = opts.parse_opt()
    opt = vars(opt)
    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)
    for a, b, c, d in train_loader:
        print(a.shape, b.shape, c.shape, d.shape)
        break
