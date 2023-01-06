import json
import os

from PIL import Image

with open('data/test_tracks.json','r') as fid:
    test_tracks = json.load(fid)

# generate test_info.json which is used in the direction prediction
test_info = dict()
for track_id in test_tracks.keys():
    track = test_tracks[track_id]
    info = track['frames'][0].split("/")
    seq_id = info[2]
    cam_id = info[3]
    img = Image.open("../data/aicity/bk_map/{}.jpg".format(seq_id+"_"+cam_id))
    boxes = track['boxes']
    points = [[b[0]+b[2]/2,b[1]+b[3]/2] for b in boxes]
    print(img.size)
    t = dict()
    t['points'] = points
    t['img_size'] = img.size
    t['cam_id'] = cam_id

    test_info[track_id] = t
with open("data/test_info.json",'w') as f:
    json.dump(test_info,f)
