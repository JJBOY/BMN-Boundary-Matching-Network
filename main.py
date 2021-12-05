import sys
from dataset import VideoDataSet
from loss_function import bmn_loss_func, get_mask
import os
import json
import torch
import torch.nn.parallel
import torch.optim as optim
import numpy as np
import opts
from models import BMN
import pandas as pd
from post_processing import BMN_post_processing
from eval import evaluation_proposal
import time
from tqdm import tqdm
# log
import logging
import wandb

sys.dont_write_bytecode = True


def train_BMN(data_loader, model, optimizer, scheduler, epoch, bm_mask):
    model.train()
    train_pemreg_loss = 0
    train_pemclr_loss = 0
    train_tem_loss = 0
    train_loss = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in tqdm(enumerate(data_loader), total=len(data_loader)):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()
        confidence_map, start, end = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()

        train_pemreg_loss += loss[2].cpu().detach().numpy()
        train_pemclr_loss += loss[3].cpu().detach().numpy()
        train_tem_loss += loss[1].cpu().detach().numpy()
        train_loss += loss[0].cpu().detach().numpy()

    print(
        "BMN training loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, train_tem_loss / (n_iter + 1),
            train_pemclr_loss / (n_iter + 1),
            train_pemreg_loss / (n_iter + 1),
            train_loss / (n_iter + 1)))
        

    return train_tem_loss / (n_iter + 1), train_pemclr_loss / (n_iter + 1), train_pemreg_loss / (n_iter + 1), train_loss / (n_iter + 1)


def validate_BMN(val_data_loader, model, epoch, bm_mask):
    model.eval()
    best_loss = 1e10
    val_pemreg_loss = 0
    val_pemclr_loss = 0
    val_tem_loss = 0
    val_loss = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in tqdm(enumerate(val_data_loader), total=len(val_data_loader)):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()

        confidence_map, start, end = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())

        val_pemreg_loss += loss[2].cpu().detach().numpy()
        val_pemclr_loss += loss[3].cpu().detach().numpy()
        val_tem_loss += loss[1].cpu().detach().numpy()
        val_loss += loss[0].cpu().detach().numpy()

    print(
        "Validation loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, val_tem_loss / (n_iter + 1),
            val_pemclr_loss / (n_iter + 1),
            val_pemreg_loss / (n_iter + 1),
            val_loss / (n_iter + 1)))


    return val_tem_loss / (n_iter + 1), val_pemclr_loss / (n_iter + 1), val_pemreg_loss / (n_iter + 1), val_loss / (n_iter + 1)


def BMN_Train(opt, reverse=False):
    # logging
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Create log file and add config opts
    logging.basicConfig(filename=os.path.join(log_dir, f'run_{time.strftime("%b%e-%H%M")}'), level=logging.WARNING)
    logging.warning(str(opt))
    # end logging
    if opt["experiment_name"] != "debug":
        wandb.init(project='11785-Project-Grp9',
                    config=opt,
                    name=opt['experiment_name'])  # init WandB

    model = BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt["training_lr"],
                           weight_decay=opt["weight_decay"])

    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train", reverse=reverse),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation", reverse=reverse),
                                              batch_size=opt["batch_size"], shuffle=False,
                                              num_workers=8, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt["patience"], factor=opt["step_gamma"], verbose=True)
    bm_mask = get_mask(opt["temporal_scale"])
    epochs = opt["train_epochs"]
    print(f"Starting training for {epochs} epochs")
    best_loss = None
    for epoch in range(epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.5f}")
        train_tem_loss, train_pemclr_loss, train_pemreg_loss, train_loss = train_BMN(train_loader, model, optimizer, scheduler, epoch, bm_mask)
        val_tem_loss, val_pemclr_loss, val_pemreg_loss, val_loss = validate_BMN(test_loader, model, epoch, bm_mask)
        scheduler.step(val_loss)

        # logging.warning("Training loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
        #     epoch, train_tem_loss,
        #     train_pemclr_loss,
        #     train_pemreg_loss,
        #     train_loss)) # log train stats

        # logging.warning("Validation loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
        #     epoch, val_tem_loss,
        #     val_pemclr_loss,
        #     val_pemreg_loss,
        #     val_loss)) # log val stats

        # wandb log
        if opt["experiment_name"] != "debug":
            wandb.log({'epoch': epoch,
                        'lr': optimizer.param_groups[0]['lr'],
                        'train_tem_loss': train_tem_loss,
                        'train_pemreg_loss': train_pemreg_loss,
                        'train_pemclr_loss': train_pemclr_loss,
                        'train_loss': train_loss,
                        'val_tem_loss': val_tem_loss,
                        'val_pemreg_loss': val_pemreg_loss,
                        'val_pemclr_loss': val_pemclr_loss,
                        'val_loss': val_loss,
                    })

        state = {'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()}
        torch.save(state, opt["checkpoint_path"] + f"/{opt['experiment_name']}_{epoch}.pth.tar")


        if not best_loss or val_loss < best_loss:
            best_loss = val_loss
            torch.save(state, opt["checkpoint_path"] + f"/{opt['experiment_name']}_BMN_best.pth.tar")


def BMN_inference(opt):
    f_model = BMN(opt)
    f_model = torch.nn.DataParallel(f_model, device_ids=[0, 1]).cuda()
    f_checkpoint = torch.load(opt["checkpoint_path"] + f"/{opt['forward_model']}_BMN_best.pth.tar")
    # r_checkpoint = torch.load(opt["checkpoint_path"] + f"/{opt['forward_model']}_BMN_best.pth.tar")

    f_model.load_state_dict(f_checkpoint['state_dict'])
    f_model.eval()
    

    if opt['ensemble'] != 0:
        r_model = BMN(opt)
        r_model = torch.nn.DataParallel(r_model, device_ids=[0, 1]).cuda()
        r_checkpoint = torch.load(opt["checkpoint_path"] + f"/{opt['reverse_model']}_BMN_best.pth.tar")
        r_model.load_state_dict(r_checkpoint['state_dict'])
        r_model.eval()


    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation", reverse=False),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)
    tscale = opt["temporal_scale"]
    with torch.no_grad():
        for idx, input_data in tqdm(test_loader, total=len(test_loader)):
            video_name = test_loader.dataset.video_list[idx[0]]
            input_data = input_data.cuda()
            confidence_map, start, end = f_model(input_data)

            start_scores = start[0].detach().cpu().numpy()
            end_scores = end[0].detach().cpu().numpy()

            if opt['ensemble'] != 0:
                input_data_time_flipped = torch.flip(input_data, dims=(2,))
                r_confidence_map, r_start, r_end = r_model(input_data_time_flipped)
                r_end = torch.flip(r_end, dims=(1,))
                r_start = torch.flip(r_start, dims=(1,))
                r_confidence_map = torch.transpose(torch.flip(r_confidence_map, dims=(2, 3)), 2, 3)

                r_start_scores = r_end[0].detach().cpu().numpy()
                r_end_scores = r_start[0].detach().cpu().numpy()

                start_scores = (start_scores + r_start_scores) / 2
                end_scores = (end_scores + r_end_scores) / 2

                clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
                reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

                r_clr_confidence = (r_confidence_map[0][1]).detach().cpu().numpy()
                r_reg_confidence = (r_confidence_map[0][0]).detach().cpu().numpy()

                clr_confidence = (clr_confidence + r_clr_confidence) / 2
                reg_confidence = (reg_confidence + r_reg_confidence) / 2
                
            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

            new_props = []
            for idx in range(tscale):
                for jdx in range(tscale):
                    start_index = idx
                    end_index = jdx + 1
                    if start_index < end_index and  end_index<tscale :
                        xmin = start_index / tscale
                        xmax = end_index / tscale
                        xmin_score = start_scores[start_index]
                        xmax_score = end_scores[end_index]
                        clr_score = clr_confidence[idx, jdx]
                        reg_score = reg_confidence[idx, jdx]
                        score = xmin_score * xmax_score * clr_score * reg_score
                        new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
            new_props = np.stack(new_props)
            #########################################################################

            col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
            new_df = pd.DataFrame(new_props, columns=col_name)
            new_df.to_csv("./output/BMN_results/" + video_name + ".csv", index=False)


def main(opt):
    if opt["mode"] == "train":
        if opt["reverse"] == 0:
            reverse = False
        else:
            reverse = True
        BMN_Train(opt, reverse=reverse)
    elif opt["mode"] == "inference":
        if not os.path.exists("output/BMN_results"):
            os.makedirs("output/BMN_results")
        BMN_inference(opt)
        print("Post processing start")
        BMN_post_processing(opt)
        print("Post processing finished")
        evaluation_proposal(opt)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file = open(opt["checkpoint_path"] + "/opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()

    # Set random seed
    np.random.seed(opt['random_seed'])
    torch.manual_seed(opt['random_seed'])

    # model = BMN(opt)
    # a = torch.randn(1, 400, 100)
    # b, c = model(a)
    # print(b.shape, c.shape)
    # print(b)
    # print(c)
    main(opt)
