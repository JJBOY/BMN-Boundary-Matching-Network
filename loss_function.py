# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F


def get_mask(tscale):
    mask = np.zeros([tscale, tscale], np.float32)
    for i in range(tscale):
        for j in range(i, tscale):
            mask[i, j] = 1
    return torch.Tensor(mask)


def bmn_loss_func(pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end, bm_mask):
    pred_bm_reg = pred_bm[:, 0].contiguous()
    pred_bm_cls = pred_bm[:, 1].contiguous()

    gt_iou_map = gt_iou_map * bm_mask

    pem_reg_loss = pem_reg_loss_func(pred_bm_reg, gt_iou_map, bm_mask)
    pem_cls_loss = pem_cls_loss_func(pred_bm_cls, gt_iou_map, bm_mask)
    tem_loss = tem_loss_func(pred_start, pred_end, gt_start, gt_end)

    loss = tem_loss + 10 * pem_reg_loss + pem_cls_loss
    return loss, tem_loss, pem_reg_loss, pem_cls_loss


def tem_loss_func(pred_start, pred_end, gt_start, gt_end):
    def bi_loss(pred_score, gt_label):
        pred_score = pred_score.view(-1)
        gt_label = gt_label.view(-1)
        pmask = (gt_label > 0.5).float()
        num_entries = len(pmask)

        epsilon = 1e-8
        num_positive = torch.sum(pmask) + epsilon
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1 + epsilon)
        coef_1 = 0.5 * ratio
        loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * (1.0 - pmask)
        loss = -1 * torch.mean(loss_pos + loss_neg)
        return loss

    loss_start = bi_loss(pred_start, gt_start)
    loss_end = bi_loss(pred_end, gt_end)
    loss = loss_start + loss_end
    return loss


def pem_reg_loss_func(pred_score, gt_iou_map, mask):
    u_hmask = (gt_iou_map > 0.7).float()
    u_mmask = ((gt_iou_map <= 0.7) & (gt_iou_map > 0.3)).float()
    u_lmask = ((gt_iou_map <= 0.3) & (gt_iou_map > 0.)).float()
    u_lmask = u_lmask * mask

    epsilon = 1e-8
    num_h = torch.sum(u_hmask) + epsilon
    num_m = torch.sum(u_mmask) + epsilon
    num_l = torch.sum(u_lmask) + epsilon

    # print(f"Num L = {num_l}, Num H = {num_h}, Num M = {num_m}")

    r_m = num_h / num_m
    u_smmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_smmask = u_mmask * u_smmask
    u_smmask = (u_smmask > (1. - r_m)).float()

    r_l = num_h / num_l
    u_slmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_slmask = u_lmask * u_slmask
    u_slmask = (u_slmask > (1. - r_l)).float()

    weights = u_hmask + u_smmask + u_slmask + epsilon

    loss = F.mse_loss(pred_score * weights, gt_iou_map * weights)
    # print(f"Weights = {weights}")
    loss = 0.5 * torch.sum(loss * torch.ones(*weights.shape).cuda()) / torch.sum(weights)
    return loss


def pem_cls_loss_func(pred_score, gt_iou_map, mask):
    pmask = (gt_iou_map > 0.9).float()
    nmask = (gt_iou_map <= 0.9).float()
    nmask = nmask * mask

    epsilon = 1e-8
    num_positive = torch.sum(pmask) + epsilon
    num_entries = num_positive + torch.sum(nmask)
    ratio = num_entries / num_positive
    coef_0 = 0.5 * ratio / (ratio - 1 + epsilon)
    coef_1 = 0.5 * ratio
    loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
    loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * nmask
    loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
    return loss
