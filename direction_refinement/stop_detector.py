import copy
import json
import os
import pdb

import cv2
import numpy as np
from PIL import Image, ImageDraw

# 1. read some data
with open('data/test_info.json','r') as fid:
    test_info = json.load(fid)
with open('data/test_tracks.json','r') as fid:
    test_tracks = json.load(fid)
with open('data/target_test_direction_predict_softmax.json','r') as fid:
    test_tracks_pred = json.load(fid)

det_imgs = json.load(open("data/test_det.json"))
det_res = json.load(open('data/track2.89.bbox.json'))
new_res = {}
new_img_size = {}
for each_res in det_res:
    image_id = each_res['image_id'] 
    det_imgs_ind = image_id - 1
    cur_img = det_imgs['images'][det_imgs_ind]
    assert cur_img['id'] == image_id
    image_name = cur_img['file_name']
    if image_name not in new_res:
        new_res[image_name] = []
    new_res[image_name].append(each_res)
    new_img_size[image_name] = [cur_img['height'], cur_img['width'], 3]


direction = ['straight', 'stop', 'left', 'right']
may_have_stop_cams = ['c001', 'c002', 'c004', 'c005', 'c030', 'c040']

# 2. some helpful functions
def get_motion_masks(boxes, uuid, img_shape):
    """Get the motion mask which is the track's movement area in the back ground image
    """
    mask = np.zeros(img_shape)
    boxes = boxes[:-10]

    for box in boxes:
        x,y,w,h = box
        x1 = int(x + w/4)
        x2 = int(x + w - w/4)
        y1 = int(y + h/4)
        y2 = int(y + h - h/4)
        mask[y1:y2, x1:x2, 0] = 1
        mask[y1:y2, x1:x2, 1] = 1
        mask[y1:y2, x1:x2, 2] = 1
    return mask

def bb_intersection_over_union(boxA, boxB):
    """Get the iou of two bounding boxes
    """
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def has_front_car(each_uuid, frame_name, ox1, oy1, ox2, oy2):
    """Check if there is a vehicle in front of the track
    """
    anno_track = test_tracks[each_uuid]['boxes']
    anno_frames = test_tracks[each_uuid]['frames']
    frame_index = anno_frames.index(frame_name)
    img_shape = new_img_size[frame_name]
    ocx = (ox1 + ox2) / 2.0
    ocy = (oy1 + oy2) / 2.0
    tgt_center = [ocx, ocy]
    next_frame = frame_index + 1
    dist_tgt = 0
    # find a distance next frame center bbox
    while dist_tgt < 100:
        bbox_gt_next = anno_track[next_frame]
        tgt_center_next = [bbox_gt_next[0] + bbox_gt_next[2]/2,bbox_gt_next[1] + bbox_gt_next[3]/2]
        dist_tgt = (tgt_center_next[0] - tgt_center[0])**2 + (tgt_center_next[1] - tgt_center[1])**2
        next_frame += 1
        if next_frame > (len(anno_track) - 1):
            break
    mask = get_motion_masks(anno_track, each_uuid, img_shape)
    cur_res = new_res[frame_name]
    track_bbox = []
    iou_max = 0.0
    # find a bbox overlap current target car trajectory
    for each_res in cur_res:
        x1, y1, w, h = each_res['bbox']
        cx = x1 + w/ 2.0 
        cy = y1 + h/ 2.0 
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x1 + w)
        y2 = int(y1 + h)
        cur_dis = (ocx - cx) * (ocx - cx) + (ocy - cy) * (ocy - cy)
        iou = bb_intersection_over_union([x1,y1,x2,y2], [ox1, oy1, ox2, oy2])
        if iou > iou_max:
            iou_max = iou
            best_box = [x1, y1, w, h]
        ratio = mask[y1:y2,x1:x2].sum()/(w*h)
        if ratio > 0.5:
            track_bbox.append([x1, y1, int(w), int(h)])
    tgt_center = [best_box[0] + best_box[2]/2,best_box[1] + best_box[3]/2]
    has_front = 0
    for bbox in track_bbox:
        center = [bbox[0] + bbox[2]/2,   bbox[1] + bbox[3]/2]
        dist = (tgt_center[0] - center[0])**2 + (tgt_center[1] - center[1])**2
        dist_next = (tgt_center_next[0] - center[0])**2 + (tgt_center_next[1] - center[1])**2
        if dist_next <= dist:
            has_front = 1
            break

    return has_front


small_dis_ratio = 0.35
center_var_thre = 7
slide_win_len = 10
truncate_len = 0.25

def has_stop(boxes, each_uuid):
    """Check if the track is stop 
    """
    nums = len(boxes)
    trun_nums = int(nums*truncate_len)
    used_nums = nums - trun_nums
    boxes = boxes[int(nums*truncate_len):]
    area = np.zeros(used_nums)
    center = np.zeros((used_nums, 2))
    for i in range(used_nums):
        bb = boxes[i]
        area[i] = bb[2] * bb[3]
        center[i, 0] = bb[0] + bb[2] / 2
        center[i, 1] = bb[1] + bb[3] / 2

    # use sliding windows to check if the car has moved during this period, check speed-likely variance
    total_windows_num = used_nums - slide_win_len + 1 
    center_var_list = []
    for aa in range(total_windows_num):
        inside_window_distance = center[aa:(aa+slide_win_len), :]
        inside_window_distance_x_var = inside_window_distance[:,0].var() 
        inside_window_distance_y_var = inside_window_distance[:,1].var() 
        center_var = inside_window_distance_x_var + inside_window_distance_y_var
        
        # un-norm case
        center_var_list.append(center_var)

    center_var_list = np.array(center_var_list)
   
    if (center_var_list < center_var_thre).sum() > nums * small_dis_ratio:
        if center_var_list[0] > center_var_list[-1]:
            return 1, "final_stop"
        else:
            
            frame_name = test_tracks[each_uuid]['frames'][0]
            cur_box = test_tracks[each_uuid]['boxes'][0]
            x1, y1, w, h = cur_box
            x2 = x1 + w
            y2 = y1 + h
            flag_has_front = has_front_car(each_uuid, frame_name, x1, y1, x2, y2)
            if not flag_has_front:
                return 1, "final_move"
            else:
                return 0, 'none'
    else:
        return 0, 'none'


# 3. the stop check funsction
motion_thres = 0.54
def check_stop(track_id):
    """Check a track is stop or not
    input : str,track_id
    output: bool, 1 stop 0 not stop
    """
    # read info aboud center points and gt anno
    info = test_info[track_id]
    cur_track_cp = info['points']
    if len(cur_track_cp) < 100 or len(cur_track_cp) > 618 :
        return 0
    cur_track_cam = info['cam_id']
    if cur_track_cam not in may_have_stop_cams:
        return 0
    cur_track_cp = np.array(cur_track_cp)

    # read motion prediction
    cur_motion_pred = np.array(test_tracks_pred[track_id])
    cur_motion_pred_prob = cur_motion_pred.max()
    if cur_motion_pred_prob <= motion_thres:
        return 0

    # read original bboxes
    all_bboxes = test_tracks[track_id]['boxes']

    # check stop or not
    pred_stop, _ = has_stop(all_bboxes, track_id)
    return pred_stop

if __name__ == "__main__":
    # check whether our function works correctly
    pred_stop_num = 0
    for track_id in test_info.keys():
        cur_pred_dir = check_stop(track_id)
        if cur_pred_dir:
            pred_stop_num += 1
    print(pred_stop_num)
