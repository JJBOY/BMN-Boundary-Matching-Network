import os
import json
import numpy as np
import pandas as pd

# Opening JSON file
# with open('activitynet_13_annotations.json') as json_file:
#     data1 = json.load(json_file)

def get_iou(start1: float, end1: float, start2: float, end2: float) -> float:
    intersection = max(max((end2-start1), 0) - max((end2-end1), 0) - max((start2-start1), 0), 0)

    if start1 > end2 or start2 > end1:
        union = (end1 - start1) + (end2 - start2)
    else:
        union = max(end1, end2) - min(start1, start2)
    return intersection / union

def get_mean_median_duration(start_id, end_id, vid_ious, vid_info): # exclusive of end_id
    durations = []
    for i in range(start_id, end_id):
        vid_id = vid_ious[i][0]
        durations.append(vid_info[vid_id]['duration'])
    mean = float(sum(durations))/ len(durations)
    sorted_durations = sorted(durations)
    n = len(durations)
    if n % 2 == 0:
        median = (sorted_durations[n//2] + sorted_durations[(n//2)+1]) / 2
    else:
        median = sorted_durations[n//2]

    return mean, median


with open('activity_net_1_3_new.json') as json_file:
    ground_truth = json.load(json_file)

vid_info = ground_truth['database']
res_dir = '/home/dl_g9/BMN-Boundary-Matching-Network/output/BMN_results'
vid_ious = []
for csv_file in os.listdir(res_dir):
    proposals = pd.read_csv(os.path.join(res_dir, csv_file))
    vid_id = csv_file[2:-4]

    duration = vid_info[vid_id]['duration']
    start, end = vid_info[vid_id]['annotations'][0]['segment'][0], vid_info[vid_id]['annotations'][0]['segment'][1]

    best_proposal = proposals[proposals['score'] == proposals['score'].max()]
    pred_start, pred_end = best_proposal['xmin'].iloc[0] * duration, best_proposal['xmax'].iloc[0] * duration
    # print(f'start: {type(pred_start)}, end:{type(pred_end)}')

    iou = get_iou(start, end, pred_start, pred_end)
    # print(iou)
    vid_ious.append((vid_id, iou))

vid_ious = sorted(vid_ious, key=lambda x: x[1]) # sort vid_ids according to iou
num_vids = len(vid_ious)

bot25_mean, bot25_median = get_mean_median_duration(0, num_vids//4, vid_ious, vid_info)
bot50_mean, bot50_median = get_mean_median_duration(0, num_vids//2, vid_ious, vid_info)
_, _ = get_mean_median_duration(num_vids//4, num_vids//2, vid_ious, vid_info)
_, _ = get_mean_median_duration(num_vids//2, 3*num_vids//4, vid_ious, vid_info)
top25_mean, top25_median = get_mean_median_duration(3*num_vids//4, num_vids, vid_ious, vid_info)

print(bot25_mean, bot25_median, bot50_mean, bot50_median, top25_mean, top25_median)

df = pd.DataFrame(vid_ious, columns=['vid_id', 'iou']) 
df.to_csv('case_study.csv')     








