import copy
import os

import numpy as np

from share_cam_dir_detector import *
from stop_detector import *
from unique_cam_dir_detector import *

# 1. load data
train_track_pth = "../data/aicity/annotations/train_tracks.json"
test_track_pth = "../data/aicity/annotations/test_tracks.json"

train_tracks = json.load(open(train_track_pth))
test_tracks = json.load(open(test_track_pth))

test_cam = []
train_cam = []
for track_id in test_tracks.keys():
    info = test_tracks[track_id]['frames'][0].split("/")
    cam_id = info[3]
    test_cam.append(cam_id)

for track_id in train_tracks.keys():
    info = train_tracks[track_id]['frames'][0].split("/")
    cam_id = info[3]
    train_cam.append(cam_id)

train_cam = list(set(train_cam))
test_cam = list(set(test_cam))
test_unique_cam = list(set(test_cam).difference(set(train_cam)))
test_share_cam = list(set(test_cam).intersection(set(train_cam)))

with open('data/test_info.json','r') as fid:
    test_info = json.load(fid)

# 2. get the direction prediction
pred_dir_dict = dict()

share_count = 0
unique_count = 0
unsolved = dict()
correct = 0
for track_id in test_info.keys():
    cam_id = test_info[track_id]['cam_id']
    if cam_id in test_share_cam:
        share_count += 1
        pred_dir = get_share_cam_dir(track_id)
        dir_label = [0,0,0,0]
        dir_label[int(pred_dir)] = 1
        pred_dir_dict[track_id] = dir_label
    else:
        # check if is stop
        is_stop = check_stop(track_id)
        if is_stop:
            pred_dir = 1 
        else:
            pred_dir = get_unique_cam_dir(track_id)
        
        unique_count += 1
        dir_label = [0,0,0,0]
        dir_label[int(pred_dir)] = 1
        pred_dir_dict[track_id] = dir_label
      

print(share_count)
print(unique_count)

with open("results/target_test_direction_predict_one_hot_refinement.json",'w') as f:
    json.dump(pred_dir_dict,f)
