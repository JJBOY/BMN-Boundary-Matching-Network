"""
This implementation largely borrows from [BSN](https://github.com/wzmsltw/BSN-boundary-sensitive-network) by Tianwei Lin.
"""
import sys
from dataset import VideoDataSet
from loss_function import TEM_loss_function, PEM_loss_function
import os
import json
import torch
import torch.nn.parallel
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
import opts
from models import BMN
import pandas as pd
from post_processing import BMN_post_processing
from eval import evaluation_proposal

sys.dont_write_bytecode = True

confidence_mask = torch.zeros(100, 100)
for s in range(100):
    for d in range(100):
        e = s + d
        if e > 99:
            break
        confidence_mask[d, s] = 1
confidence_mask = confidence_mask.cuda()


def train_BMN(data_loader, model, optimizer, epoch, writer, opt):
    model.train()
    epoch_pem_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, label_start, label_end, label_confidence) in enumerate(data_loader):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()

        start_end, confidence_map = model(input_data)
        tem_loss = TEM_loss_function(label_start, label_end, start_end, opt)
        pem_loss = PEM_loss_function(label_confidence, confidence_map, confidence_mask, opt)
        loss = tem_loss + pem_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_pem_loss += pem_loss.cpu().detach().numpy()
        epoch_tem_loss += tem_loss.cpu().detach().numpy()
        epoch_loss += loss.cpu().detach().numpy()

    writer.add_scalars('data/pem_loss', {'train': epoch_pem_loss / (n_iter + 1)}, epoch)
    writer.add_scalars('data/tem_loss', {'train': epoch_tem_loss / (n_iter + 1)}, epoch)
    writer.add_scalars('data/total_loss', {'train': epoch_loss / (n_iter + 1)}, epoch)

    print("BMN training loss(epoch %d): tem_loss: %.03f, pem_loss: %.03f, total_loss: %.03f" % (
        epoch, epoch_tem_loss / (n_iter + 1),
        epoch_pem_loss / (n_iter + 1),
        epoch_loss / (n_iter + 1)))


def test_BMN(data_loader, model, epoch, writer, opt):
    model.eval()
    epoch_pem_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, label_start, label_end, label_confidence) in enumerate(data_loader):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()

        start_end, confidence_map = model(input_data)
        tem_loss = TEM_loss_function(label_start, label_end, start_end, opt)
        pem_loss = PEM_loss_function(label_confidence, confidence_map, confidence_mask, opt)
        loss = tem_loss + pem_loss

        epoch_pem_loss += pem_loss.cpu().detach().numpy()
        epoch_tem_loss += tem_loss.cpu().detach().numpy()
        epoch_loss += loss.cpu().detach().numpy()

    writer.add_scalars('data/pem_loss', {'train': epoch_pem_loss / (n_iter + 1)}, epoch)
    writer.add_scalars('data/tem_loss', {'train': epoch_tem_loss / (n_iter + 1)}, epoch)
    writer.add_scalars('data/total_loss', {'train': epoch_loss / (n_iter + 1)}, epoch)

    print("BMN testing loss(epoch %d): tem_loss: %.03f, pem_loss: %.03f, total_loss: %.03f" % (
        epoch, epoch_tem_loss / (n_iter + 1),
        epoch_pem_loss / (n_iter + 1),
        epoch_loss / (n_iter + 1)))

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"] + "/BMN_checkpoint.pth.tar")
    if epoch_loss < model.best_loss:
        model.best_loss = epoch_loss
        torch.save(state, opt["checkpoint_path"] + "/BMN_best.pth.tar")


def BMN_Train(opt):
    writer = SummaryWriter()
    model = BMN(opt).cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt["training_lr"], weight_decay=opt["weight_decay"])

    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=opt["batch_size"], shuffle=False,
                                              num_workers=8, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])

    for epoch in range(opt["train_epochs"]):
        scheduler.step()
        train_BMN(train_loader, model, optimizer, epoch, writer, opt)
        test_BMN(test_loader, model, epoch, writer, opt)
    writer.close()


def BMN_inference(opt):
    model = BMN(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/BMN_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)
    tscale = opt["temporal_scale"]
    tgap = 1. / tscale
    peak_thres = opt["pgm_threshold"]
    with torch.no_grad():
        for idx, input_data in test_loader:
            video_name = test_loader.dataset.video_list[idx[0]]
            input_data = input_data.cuda()
            start_end, confidence_map = model(input_data)

            start_scores = start_end[0][0].detach().cpu().numpy()
            end_scores = start_end[0][1].detach().cpu().numpy()
            clr_confidence = (confidence_map[0][0] * confidence_mask).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][1] * confidence_mask).detach().cpu().numpy()

            max_start = max(start_scores)
            max_end = max(end_scores)

            ####################################################################################################
            # generate the set of start points and end points
            start_bins = np.zeros(len(start_scores))
            start_bins[[0, -1]] = 1  # [1,0,0...,0,1] 首末两帧
            for idx in range(1, tscale - 1):
                if start_scores[idx] > start_scores[idx + 1] and start_scores[idx] > start_scores[idx - 1]:
                    start_bins[idx] = 1
                elif start_scores[idx] > (peak_thres * max_start):
                    start_bins[idx] = 1

            end_bins = np.zeros(len(end_scores))
            end_bins[[0, -1]] = 1
            for idx in range(1, tscale - 1):
                if end_scores[idx] > end_scores[idx + 1] and end_scores[idx] > end_scores[idx - 1]:
                    end_bins[idx] = 1
                elif end_scores[idx] > (peak_thres * max_end):
                    end_bins[idx] = 1
            ########################################################################################################

            xmin_list = []
            xmin_score_list = []
            xmax_list = []
            xmax_score_list = []
            for j in range(tscale):
                if start_bins[j] == 1:
                    xmin_list.append(tgap / 2 + tgap * j)  # [0.01,0.02]与gt的重合度高，那么实际上区间的中点才是分界点
                    xmin_score_list.append(start_scores[j])
                if end_bins[j] == 1:
                    xmax_list.append(tgap / 2 + tgap * j)
                    xmax_score_list.append(end_scores[j])

            #########################################################################
            # 遍历起始分界点与结束分界点的组合
            new_props = []
            for ii in range(len(xmax_list)):
                tmp_xmax = xmax_list[ii]
                tmp_xmax_score = xmax_score_list[ii]
                for ij in range(len(xmin_list)):
                    tmp_xmin = xmin_list[ij]
                    tmp_xmin_score = xmin_score_list[ij]
                    if tmp_xmin >= tmp_xmax:
                        break
                    start_point = int((tmp_xmin - tgap / 2) / tgap)
                    end_point = int((tmp_xmax - tgap / 2) / tgap)
                    duration = end_point - start_point
                    clr_score = clr_confidence[duration, start_point]
                    reg_score = reg_confidence[duration, start_point]
                    score = tmp_xmax_score * tmp_xmax_score * np.sqrt(clr_score * reg_score)
                    if score == 0:
                        print(video_name, tmp_xmin, tmp_xmax, tmp_xmin_score, tmp_xmax_score, clr_score, reg_score,
                              score, confidence_map[0, 0, duration, start_point], duration, start_point)
                    new_props.append([tmp_xmin, tmp_xmax, tmp_xmin_score, tmp_xmax_score, clr_score, reg_score, score])
            new_props = np.stack(new_props)
            #########################################################################

            col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
            new_df = pd.DataFrame(new_props, columns=col_name)
            new_df.to_csv("./output/BMN_results/" + video_name + ".csv", index=False)


def main(opt):
    if opt["module"] == "BMN":
        mask = np.load(opt["bm_mask_path"])
        opt["bm_mask"] = mask
        if opt["mode"] == "train":
            BMN_Train(opt)
        elif opt["mode"] == "inference":
            if not os.path.exists("output/BMN_results"):
                os.makedirs("output/BMN_results")
            BMN_inference(opt)
        else:
            print("Wrong mode. BMN has two modes: train and inference")

    elif opt["module"] == "Post_processing":
        print("Post processing start")
        BMN_post_processing(opt)
        print("Post processing finished")

    elif opt["module"] == "Evaluation":
        evaluation_proposal(opt)
    print("")


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file = open(opt["checkpoint_path"] + "/opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()

    # model = BMN(opt)
    # a = torch.randn(1, 400, 100)
    # b, c = model(a)
    # print(b.shape, c.shape)
    # print(b)
    # print(c)
    main(opt)
