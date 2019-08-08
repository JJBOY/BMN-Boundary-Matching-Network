# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F


def bi_loss(scores, anchors, opt):
    scores = scores.view(-1).cuda()
    anchors = anchors.contiguous().view(-1)

    pmask = (scores > opt["tem_match_thres"]).float().cuda()
    num_positive = torch.sum(pmask)
    num_entries = len(scores)
    ratio = num_entries / num_positive

    coef_0 = 0.5 * (ratio) / (ratio - 1)
    coef_1 = coef_0 * (ratio - 1)
    loss = coef_1 * pmask * torch.log(anchors + 0.00001) + coef_0 * (1.0 - pmask) * torch.log(1.0 - anchors + 0.00001)
    loss = -torch.mean(loss)
    num_sample = [torch.sum(pmask), ratio]
    return loss, num_sample


def TEM_loss_calc(anchors_start, anchors_end,
                  match_scores_start, match_scores_end, opt):
    loss_start_small, num_sample_start_small = bi_loss(match_scores_start, anchors_start, opt)
    loss_end_small, num_sample_end_small = bi_loss(match_scores_end, anchors_end, opt)

    loss_dict = {"loss_start": loss_start_small, "num_sample_start": num_sample_start_small,
                 "loss_end": loss_end_small, "num_sample_end": num_sample_end_small}
    return loss_dict


def TEM_loss_function(y_start, y_end, TEM_output, opt):
    anchors_start = TEM_output[:, 0, :]
    anchors_end = TEM_output[:, 1, :]
    loss_dict = TEM_loss_calc(anchors_start, anchors_end,
                              y_start, y_end, opt)

    cost = loss_dict["loss_start"] + loss_dict["loss_end"]
    loss_dict["cost"] = cost
    return cost


def bi_loss_2(scores, anchors, opt, num_entries):
    # because the redundancy of the confidence map,
    # the half of the data in the map is 0 and is useless,
    # they dont neither belong to positive nor negative sample

    scores = scores.view(-1).cuda()
    anchors = anchors.contiguous().view(-1)

    pmask = (scores > opt["tem_match_thres"]).float().cuda()
    num_positive = torch.sum(pmask)
    ratio = num_entries / num_positive

    coef_0 = 0.5 * (ratio) / (ratio - 1)
    coef_1 = coef_0 * (ratio - 1)
    loss = coef_1 * pmask * torch.log(anchors + 0.00001) + coef_0 * (1.0 - pmask) * torch.log(1.0 - anchors + 0.00001)
    loss = -torch.mean(loss)
    return loss


def PEM_loss_function(match_iou, anchors_iou, confidence_mask, opt):
    match_iou = match_iou.view(-1)
    clr_anchors = anchors_iou[:, 0].contiguous() * confidence_mask
    reg_anchors = anchors_iou[:, 1].contiguous() * confidence_mask
    confidence_mask = confidence_mask.view(-1)

    u_hmask = (match_iou > opt["pem_high_iou_thres"]).float()
    u_mmask = ((match_iou <= opt["pem_high_iou_thres"]) & (match_iou > opt["pem_low_iou_thres"])).float()
    u_lmask = (match_iou < opt["pem_low_iou_thres"]).float() * confidence_mask.repeat(anchors_iou.size(0))

    num_h = torch.sum(u_hmask)
    num_m = torch.sum(u_mmask)
    num_l = torch.sum(u_lmask)

    r_m = opt["pem_u_ratio_m"] * num_h / (num_m)
    r_m = torch.min(r_m, torch.Tensor([1.0]).cuda())[0]
    u_smmask = torch.Tensor(np.random.rand(u_hmask.size()[0])).cuda()
    u_smmask = u_smmask * u_mmask
    u_smmask = (u_smmask > (1. - r_m)).float()

    r_l = opt["pem_u_ratio_l"] * num_h / (num_l)
    r_l = torch.min(r_l, torch.Tensor([1.0]).cuda())[0]
    u_slmask = torch.Tensor(np.random.rand(u_hmask.size()[0])).cuda()
    u_slmask = u_slmask * u_lmask
    u_slmask = (u_slmask > (1. - r_l)).float()

    iou_weights = u_hmask + u_smmask + u_slmask
    reg_loss = F.smooth_l1_loss(reg_anchors.view(-1)*iou_weights, match_iou*iou_weights)
    reg_loss = torch.sum(reg_loss * confidence_mask) / torch.sum(iou_weights)

    num_entry = torch.sum(confidence_mask) * anchors_iou.size(0)
    clr_loss = bi_loss_2(match_iou, clr_anchors, opt, num_entry)
    #print(clr_loss,50* reg_loss)
    loss = clr_loss + 50 * reg_loss
    return loss
