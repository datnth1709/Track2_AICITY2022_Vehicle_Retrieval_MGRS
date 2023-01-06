import json
import os
import pdb

import cv2
import numpy as np
from PIL import Image

# 1. define the unique camera's direction data
logit = {
"c001":{"12":3,"13":0,"14":2,"21":2,"23":3,"24":0,"31":0,"32":2,"34":3,"41":3,"42":0,"43":2},
"c002":{"12":3,"13":0,"14":2,"21":2,"23":3,"24":0,"31":0,"32":2,"34":3,"41":3,"42":0,"43":2},
"c003":{"12":0,"13":2,"21":0,"23":3,"31":3,"32":2},
"c004":{"12":3,"13":0,"15":2,"16":2,"54":0,"64":3},
"c005":{"13":0,"14":2,"32":2,"41":3,"43":2},
"c012":{"12":2},
"c013":{"23":3,"34":3},
"c014":{"12":0,"21":0},
"c030":{"23":3,"31":0,"32":2},
"c032":{"12":0,"21":0},
"c037":{"12":0,"21":0},
"c038":{"12":0,"21":0},
"c040":{"12":0,"13":2,"21":0,"23":3,"31":3,"32":2}
}

# 2. get the camera of the test data
test_info_pth = "./data/test_info.json"
test_info = json.load(open(test_info_pth))

test_cam_unique = ['c001', 'c002', 'c003', 'c004', 'c005', 'c012', 'c013', 'c014', 'c030', 'c032', 'c037', 'c038', 'c040']

def get_cross_angle(arr_a,arr_b):
    cos_value = (float(arr_a.dot(arr_b)) / (np.sqrt(arr_a.dot(arr_a)) * np.sqrt(arr_b.dot(arr_b))))  # 注意转成浮点数运算
    return np.arccos(cos_value) * (180 / np.pi)  # 两个向量的夹角的角度， 余弦值：cos_value, np.cos(para), 其中para是弧度，不是角度

unsolved = dict()
# 3. define a function to return the predicted direction for the tracks under the overlapped camera
def get_unique_cam_dir(track_id):
    test_track = test_info[track_id]
    cam_id = test_track['cam_id']
    points = test_track['points']
    assert cam_id in test_cam_unique,"Track uuid Error! Only track in the unique cameras can get direction by deal_unique_cam_dir!!"
    # check the region
    start_p = points[0]
    end_p = points[-1]
    # read the cam road mask
    cam_mask = Image.open("data/unique_masks/{}.png".format(cam_id))
    cam_mask = np.array(cam_mask)

    # get the start_id and end_id
    start_id = None
    end_id = None
    for i in range(6):
        start_mask = -1*np.ones_like(cam_mask)
        end_mask = -1*np.ones_like(cam_mask)
        if start_id is None:
            start_mask[int(start_p[1]),int(start_p[0])] = i+1
            if np.sum(start_mask==cam_mask) != 0:
                start_id = str(i+1)
        if end_id is None:
            end_mask[int(end_p[1]),int(end_p[0])] = i+1
            if np.sum(end_mask==cam_mask) != 0:
                end_id = str(i+1)
    
    # get the angle of the points to get nearest mask area
    # original vector
    if start_id is None:
        pdb.set_trace()
        cam_regions = json.load(open("data/unique_masks/{}.json".format(cam_id)))['shapes']
        vec1 = np.array(points[0])-np.array(points[1])
        min_angle = np.Inf
        for region in cam_regions:
            r_id = region['label'].split("_")[-1]
            ct = np.array(region['points']).mean(axis=0)
            vec2 = ct - np.array(points[1])
            # calculate the angle
            angle = get_cross_angle(vec1,vec2)
            if angle < min_angle:
                min_angle = angle
                start_id = r_id
    if end_id is None:
        cam_regions = json.load(open("data/unique_masks/{}.json".format(cam_id)))['shapes']
        vec1 = np.array(points[-1])-np.array(points[-2])
        min_angle = np.Inf
        for region in cam_regions:
            r_id = region['label'].split("_")[-1]
            ct = np.array(region['points']).mean(axis=0)
            vec2 = ct - np.array(points[-2])
            # calculate the angle
            angle = get_cross_angle(vec1,vec2)
            if angle < min_angle:
                min_angle = angle
                end_id = r_id
    # pdb.set_trace();
    # get the pred direction
    if start_id and end_id:
        pred_dir = 0
        if start_id == end_id:
            return pred_dir
        try:
            pred_dir = logit[cam_id][start_id+end_id]
            return pred_dir
        except:
            pdb.set_trace()
    else:
        pdb.set_trace()

if __name__ == "__main__":
    count = 0
    for track_id in test_info.keys():
        cam_id = test_info[track_id]['cam_id']
        if cam_id in test_cam_unique:
            count += 1 
            pred_dir = deal_unique_cam_dir(track_id)
           
    print(count)

