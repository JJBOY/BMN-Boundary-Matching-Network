# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import torch.utils.data as data
import torch


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


class VideoDataSet(data.Dataset):
    def __init__(self, opt, subset="train"):
        self.temporal_scale = opt["temporal_scale"]  # 100
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = opt["mode"]
        self.feature_path = opt["feature_path"]
        self.boundary_ratio = opt["boundary_ratio"]
        self.video_info_path = opt["video_info"]
        self.video_anno_path = opt["video_anno"]
        self._getDatasetDict()

    def _getDatasetDict(self):
        anno_df = pd.read_csv(self.video_info_path)
        anno_database = load_json(self.video_anno_path)
        self.video_dict = {}
        for i in range(len(anno_df)):
            video_name = anno_df.video.values[i]
            video_info = anno_database[video_name]
            video_subset = anno_df.subset.values[i]
            if self.subset == "full":  # to generate data for next stage in BSN, but not used in BMN
                self.video_dict[video_name] = video_info
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
        self.video_list = list(self.video_dict.keys())
        print("%s subset video numbers: %d" % (self.subset, len(self.video_list)))

    def __getitem__(self, index):
        video_data, anchor_xmin, anchor_xmax = self._get_base_data(index)
        if self.mode == "train":
            match_score_start, match_score_end, confidence_score = self._get_train_label(index, anchor_xmin,
                                                                                         anchor_xmax)
            return video_data, match_score_start, match_score_end, confidence_score
        else:
            return index, video_data

    def _get_base_data(self, index):
        video_name = self.video_list[index]
        anchor_xmin = [self.temporal_gap * i for i in range(self.temporal_scale)]  # [0, 0.99]
        anchor_xmax = [self.temporal_gap * i for i in range(1, self.temporal_scale + 1)]  # [0.01, 1]
        video_df = pd.read_csv(self.feature_path + "csv_mean_" + str(self.temporal_scale) + "/" + video_name + ".csv")
        video_data = video_df.values[:, :]
        video_data = torch.Tensor(video_data)
        video_data = torch.transpose(video_data, 0, 1)
        video_data.float()
        return video_data, anchor_xmin, anchor_xmax

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
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])
        ##############################################################################################

        ####################################################################################################
        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        #####################################################################################################

        ##########################################################################################################
        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        ############################################################################################################

        ############################################################################################################
        # calculate the confidence(IoU) for all possible proposal
        match_score_confidence = np.zeros([self.temporal_scale, self.temporal_scale])
        for s in range(self.temporal_scale):
            for d in range(1, self.temporal_scale):
                e = s + d
                if e > self.temporal_scale - 1:
                    break
                tstart = s / self.temporal_scale + self.temporal_gap / 2
                tend = e / self.temporal_scale + self.temporal_gap / 2
                twidth = tend - tstart
                match_score_confidence[d, s] = np.max(self._iou_with_anchors(tstart, tend, twidth, gt_xmins, gt_xmaxs))
        match_score_confidence = torch.Tensor(match_score_confidence)
        ##############################################################################################################

        return match_score_start, match_score_end, match_score_confidence

    def _ioa_with_anchors(self, anchors_min, anchors_max, box_min, box_max):
        # calculate the overlap proportion between the anchor and all bbox for supervise signal,
        # the length of the anchor is 0.01
        len_anchors = anchors_max - anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        scores = np.divide(inter_len, len_anchors)  # a point to all gt
        return scores

    def _iou_with_anchors(self, anchors_min, anchors_max, len_anchors, box_min, box_max):
        """Compute jaccard score between a box and the anchors.
        """
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        union_len = len_anchors - inter_len + box_max - box_min
        # print(len_anchors,inter_len,union_len)
        jaccard = np.divide(inter_len, union_len)
        return jaccard

    def __len__(self):
        return len(self.video_list)
