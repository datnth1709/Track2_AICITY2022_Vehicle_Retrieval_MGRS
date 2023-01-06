import json
import os

import numpy as np
from PIL import Image, ImageDraw

cam_imgs = os.listdir("data/imgs/")
for img_name in cam_imgs:
    cam_img = Image.open("data/imgs/"+img_name)
    cam_draw = ImageDraw.Draw(cam_img,'RGBA')
    cam_id = img_name.split(".")[0]

    fill_cols = [(255,0,0,70),(0,255,0,70),(0,0,255,70),(255,0,255,70),(0,255,255,70),(255,255,0,70)]
    region_dict = json.load(open("data/masks/{}.json".format(cam_id)))
    for r_dict in region_dict['shapes']:
        # draw the polygon
        points = r_dict['points']
        points = [(p[0],p[1]) for p in points]
        cam_draw.polygon(points,fill =fill_cols[int(r_dict['label'].split("_")[-1])-1],outline='black')
    cam_img.save("road_mask/draw_region_{}.jpg".format(cam_id))
