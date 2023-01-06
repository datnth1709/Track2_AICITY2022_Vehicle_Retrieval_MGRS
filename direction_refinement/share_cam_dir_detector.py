import json
import pdb

import numpy as np
from PIL import Image

# 1. define the overlapped camera's direction data
# overlapped camera's direction is dicide by their road
logit = {
"c010":{"12":3,"13":0,"14":2,"21":2,"23":3,"24":0,"31":0,"32":2,"34":3,"41":3,"42":0,"43":2},
"c016":{"13":0,"23":0,"31":0,"32":0},
"c017":{"13":0,"12":0,"21":0,"23":2,"31":0,"32":3},
"c019":{"12":2,"13":3,"14":0,"21":3,"23":2,"24":2,"34":0,"31":2,"32":2,"41":3,"42":3,"43":0},
"c020":{"12":0,"21":0},
"c021":{"12":0,"21":0},
"c022":{"12":0,"21":0},
"c025":{"12":0,"21":0},
"c026":{"12":3,"13":0,"14":2,"21":2,"23":3,"24":0,"31":0,"32":2,"34":3,"41":3,"42":0,"43":2},
"c027":{"12":3,"13":0,"14":2,"21":2,"23":3,"24":0,"31":0,"32":2,"34":3,"41":3,"42":0,"43":2},
"c029":{"12":3,"13":0,"21":2,"23":3,"31":0,"32":2},
"c033":{"12":2,"13":3,"14":0,"21":3,"23":2,"24":2,"31":2,"32":3,"34":3,"41":0,"42":3,"43":2},
"c034":{"12":0,"21":0},
"c035":{"12":3,"13":2,"21":2,"23":3,"31":3,"32":2},
"c036":{"12":3,"13":0,"14":2,"21":2,"23":3,"24":3,"31":0,"32":2,"34":3,"41":3,"42":3,"43":2}}

# 2. get the camera of the test data
test_info_pth = "./data/test_info.json"
test_info = json.load(open(test_info_pth))

test_cam_unique = ['c001', 'c002', 'c003', 'c004', 'c005', 'c012', 'c013', 'c014', 'c030', 'c032', 'c037', 'c038', 'c040']

# 3. define a function to return the predicted direction for the tracks under the overlapped camera
def get_share_cam_dir(track_id):
    test_track = test_info[track_id]
    cam_id = test_track['cam_id']
    points = test_track['points']
    assert cam_id not in test_cam_unique,"Track uuid Error! Only track in the overlapped cameras can get direction by get_share_cam_dir!!"
    # check the region
    start_p = points[0]
    end_p = points[-1]
    # read the cam road mask
    cam_mask = Image.open("data/masks/{}.png".format(cam_id))
    cam_mask = np.array(cam_mask)
    # get the start_id and end_id
    start_id = None
    end_id = None
    for i in range(4):
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
   


if __name__ == "__main__":
    count = 0
    for track_id in test_info.keys():
        cam_id = test_info[track_id]['cam_id']
        if cam_id not in test_cam_unique:
            count += 1 
            pred_dir = get_share_cam_dir(track_id)
    print(count)
