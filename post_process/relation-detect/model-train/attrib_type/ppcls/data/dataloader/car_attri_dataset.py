#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import os
import cv2

from ppcls.data.preprocess import transform
from ppcls.utils import logger

from .common_dataset import CommonDataset, create_operators


class CarAttriDataset(CommonDataset):
    def __init__(
            self,
            image_root,
            cls_label_path,
            is_multi_head=False,
            transform_ops=None, ):
        self._img_root = image_root
        self._cls_path = cls_label_path
        self._is_multi_head = is_multi_head
        if transform_ops:
            self._transform_ops = create_operators(transform_ops)

        self.images = []
        self.labels = []
        self._load_anno()

    def _load_anno(self):

        assert os.path.exists(self._cls_path)
        assert os.path.exists(self._img_root)
        self.images = []
        self.labels = []
        with open(self._cls_path) as fd:
            lines = fd.readlines()
            for line in lines:
                line = line.strip('\n').split("\t")
                img_path = line[0]
                labels = [int(v) for v in line[1:]]
                image_path = os.path.join(self._img_root, img_path)
#                 print('image_path: ',image_path)
                if not os.path.exists(image_path):
                    print('file is not exit:',image_path)
                    continue
                self.images.append(image_path)
                self.labels.append(labels)
                

    def __getitem__(self, idx):
        try:
            with open(self.images[idx], 'rb') as f:
                img = f.read()
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            img = img.transpose((2, 0, 1))
            label = np.array(self.labels[idx])
#             assert len(label) == 8
                
            return (img, label)
        
#             if not self._is_multi_head:
#                 label = np.array(self.labels[idx])
#                 return (img, label)
#             else:
#                 car_label = np.array(self.labels[idx][0])
#                 year_label = np.array(self.labels[idx][1])
#                 color_label = np.array(self.labels[idx][2])
#                 brand_label = np.array(self.labels[idx][3])
#                 return (img, car_label, year_label, color_label, brand_label)

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[idx], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)
