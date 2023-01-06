# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

# from ppcls.data.preprocess.ops.autoaugment import ImageNetPolicy as RawImageNetPolicy
from ppcls.data.preprocess.ops.autoaugment import CarAttriPolicy as RawImageNetPolicy
from ppcls.data.preprocess.ops.randaugment import RandAugment as RawRandAugment
from ppcls.data.preprocess.ops.timm_autoaugment import RawTimmAutoAugment
from ppcls.data.preprocess.ops.cutout import Cutout

from ppcls.data.preprocess.ops.hide_and_seek import HideAndSeek
from ppcls.data.preprocess.ops.random_erasing import RandomErasing
from ppcls.data.preprocess.ops.grid import GridMask

from ppcls.data.preprocess.ops.operators import DecodeImage
from ppcls.data.preprocess.ops.operators import ResizeImage
from ppcls.data.preprocess.ops.operators import CropImage
from ppcls.data.preprocess.ops.operators import RandCropImage
from ppcls.data.preprocess.ops.operators import RandFlipImage
from ppcls.data.preprocess.ops.operators import NormalizeImage
from ppcls.data.preprocess.ops.operators import ToCHWImage
from ppcls.data.preprocess.ops.operators import AugMix

from ppcls.data.preprocess.batch_ops.batch_operators import MixupOperator, CutmixOperator, OpSampler, FmixOperator
import imgaug.augmenters as iaa
import imgaug as ia
import numpy as np
from PIL import Image


def transform(data, ops=[]):
    """ transform """
    for op in ops:
        data = op(data)
    return data

class IaaPolicy():
    def __init__(self, *args, **kwargs):
        pass
        
    def __call__(self, img):
#         print("do IaaPolicy")
        self.seq = iaa.Sequential(
            [iaa.SomeOf((0, 2),[
                # crop images by -5% to 10% of their height/width
#                 iaa.CropAndPad(
#                     percent=(-0.05, 0.1),
#                     pad_mode=ia.ALL,
#                     pad_cval=(0, 255)
#                 ),

                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.MotionBlur(5),
                ]),

                #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.1), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                #iaa.Invert(0.01, per_channel=True), # invert color channels
                iaa.Multiply((0.5,1.5)), # change brightness of images 
                #iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                #iaa.Grayscale(alpha=(0.0, 1.0)),
                iaa.GammaContrast((0.5,1.5))
            ])
        ])
            
        if not isinstance(img, Image.Image):
            img = np.ascontiguousarray(img)
            img = Image.fromarray(img)
        
        img = np.array(img)
        img = self.seq(images=[img])[0]
        if isinstance(img, Image.Image):
            img = np.asarray(img)
            
        return img
        

class AutoAugment(RawImageNetPolicy):
    """ ImageNetPolicy wrapper to auto fit different img types """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = np.ascontiguousarray(img)
            img = Image.fromarray(img)

        img = super().__call__(img)

        if isinstance(img, Image.Image):
            img = np.asarray(img)

        return img


class RandAugment(RawRandAugment):
    """ RandAugment wrapper to auto fit different img types """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = np.ascontiguousarray(img)
            img = Image.fromarray(img)

        img = super().__call__(img)

        if isinstance(img, Image.Image):
            img = np.asarray(img)

        return img


class TimmAutoAugment(RawTimmAutoAugment):
    """ TimmAutoAugment wrapper to auto fit different img tyeps. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = np.ascontiguousarray(img)
            img = Image.fromarray(img)

        img = super().__call__(img)

        if isinstance(img, Image.Image):
            img = np.asarray(img)

        return img
