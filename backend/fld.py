"""Helper functions to support face landmark detection using
python_facelandmark_detection project"""
from typing import List, Tuple, Dict
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import cv2

# from .retinaface.core import Retinaface
from .flm_models.basenet import MobileNet_GDConv
from .utils import BBox, drawLandmark_multiple


PICS_PATH = Path( '/home/teo/Dokumente/Personales/Photos' )
FLD_PATH = Path( '/home/teo/git/pytorch_face_landmark' )

Array = np.ndarray

OUT_SIZE = 224  # for mobile net
MEAN = np.asarray([ 0.485, 0.456, 0.406 ])
STD = np.asarray([ 0.229, 0.224, 0.225 ])

# Retinaface.trained_model = str( FLD_PATH / 'Retinaface/weights/mobilenet0.25_Final.pth' )

# download model from
# https://drive.google.com/file/d/1Le5UdpMkKOTRr1sTp4lwkw8263sbgdSe/view?usp=sharing
MOBILE_NET_MODEL_FP = FLD_PATH / 'checkpoint/mobilenet_224_model_best_gdconv_external.pth.tar'

# %%


def load_landmarks_model():
    model = MobileNet_GDConv(136)
    model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    print("map_location: ", map_location)

    checkpoint = torch.load( MOBILE_NET_MODEL_FP, map_location=map_location)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def process_faces(faces: List[Array], img: np.ndarray, model: nn.Module) -> np.ndarray:
    """detect all faces in an image and their landmarks and draw bounding boxes and points"""
    for k, face in enumerate(faces):
        landmark, new_bbox = process_1_face(face, img, model)
        img = drawLandmark_multiple(img, new_bbox, landmark)

    return img


def process_1_face(face_bbox: Array, img: np.ndarray, model: nn.Module ) -> Tuple[Array, BBox]:
    """Detect facelandmarks on a face from an image"""
    new_bbox, bbox_dict = build_bbox(face_bbox, img.shape)

    dx, dy, edx, edy = bbox_dict['dx'], bbox_dict['dy'], bbox_dict['edx'], bbox_dict['edy']

    cropped = img[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]
    if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
        cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx),
                                     cv2.BORDER_CONSTANT, 0)
    cropped_face = cv2.resize(cropped, (OUT_SIZE, OUT_SIZE))

    if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
        return

    test_face = cropped_face.copy()
    test_face = test_face / 255.0
    # if args.backbone == 'MobileNet':
    test_face = (test_face - MEAN) / STD
    test_face = test_face.transpose((2, 0, 1))
    test_face = test_face.reshape((1,) + test_face.shape)
    nn_input = torch.from_numpy(test_face).float()
    print(nn_input.dtype)
    nn_input = torch.autograd.Variable(nn_input)
    start = time.time()
    # if args.backbone == 'MobileFaceNet':
    # landmark = model(input)[0].cpu().data.numpy()
    # else:
    landmark = model(nn_input).cpu().data.numpy()
    end = time.time()
    print('Time: {:.6f}s.'.format(end - start))

    landmark = landmark.reshape(-1, 2)
    landmark = new_bbox.reprojectLandmark(landmark)

    return landmark, new_bbox


def build_bbox(face: Array, img_shape: Tuple[int, int]) -> Tuple[BBox, Dict]:
    x1 = face[0]
    y1 = face[1]
    x2 = face[2]
    y2 = face[3]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    size = int(min([w, h]) * 1.2)
    cx = x1 + w // 2
    cy = y1 + h // 2
    x1 = cx - size // 2
    x2 = x1 + size
    y1 = cy - size // 2
    y2 = y1 + size

    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    # print( f'build_bbox: img_shape={img_shape}')
    height, width, _ = img_shape
    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)
    new_bbox = list(map(int, [x1, x2, y1, y2]))

    new_bbox = BBox(new_bbox)

    return new_bbox, dict(dx=dx, dy=dy, edx=edx, edy=edy)


TRIANGLES = [
    [60, 59, 67],
    [67, 59, 58],
    [67, 58, 66],
    [66, 58, 57],
    [66, 57, 56],
    [66, 56, 65],
    [48, 60, 49],
    [60, 61, 49],
    [61, 50, 49],
    [50, 61, 51],
    [51, 61, 62],
    [51, 62, 63],
    [51, 63, 52],
    [53, 64, 54],
    [53, 63, 64],
    [60, 67, 61],
    [61, 67, 62],
    [62, 67, 66],
    [62, 66, 65],
    [62, 65, 63],
    [64, 63, 65],
    [48, 59, 60],
    [54, 64, 55],
    [64, 65, 55],
    [55, 65, 56],
    [34, 52, 53],
    [34, 54, 35],
    [34, 53, 54],
    [36, 41, 37],
    [37, 41, 40],
    [37, 40, 38],
    [38, 40, 39],
    [28, 29, 31],
    [31, 40, 41],
    [31, 39, 40],
    [31, 28, 39],
    [30, 31, 32],
    [30, 32, 33],
    [30, 33, 34],
    [30, 34, 35],
    [31, 48, 32],
    [32, 48, 49],
    [32, 49, 50],
    [32, 50, 33],
    [33, 50, 51],
    [33, 51, 52],
    [33, 34, 52],
    [52, 63, 53],
    [0, 36, 17],
    [17, 36, 37],
    [17, 37, 18],
    [18, 37, 19],
    [37, 38, 19],
    [19, 38, 20],
    [20, 38, 21],
    [21, 38, 39],
    [21, 39, 27],
    [21, 22, 27],
    [27, 39, 28],
    [29, 31, 30],
    [22, 43, 42],
    [22, 27, 42],
    [27, 42, 28],
    [28, 29, 35],
    [29, 30, 35],
    [28, 35, 42],
    [42, 47, 35],
    [47, 46, 35],
    [42, 43, 47],
    [43, 47, 46],
    [43, 46, 44],
    [44, 46, 45],
    [23, 22, 43],
    [24, 23, 43],
    [24, 43, 44],
    [25, 24, 44],
    [44, 26, 25],
    [26, 44, 45],
    [16, 26, 45],
    [15, 16, 45],
    [15, 46, 45],
    [14, 15, 46],
    [14, 35, 46],
    [0, 1, 36],
    [1, 36, 41],
    [1, 41, 2],
    [2, 41, 31],
    [2, 31, 3],
    [3, 31, 48],
    [3, 48, 4],
    [4, 48, 5],
    [5, 48, 59],
    [5, 59, 6],
    [6, 59, 58],
    [6, 58, 7],
    [7, 58, 57],
    [7, 57, 8],
    [8, 57, 56],
    [8, 56, 9],
    [9, 56, 55],
    [9, 55, 10],
    [10, 55, 54],
    [10, 11, 54],
    [11, 54, 12],
    [12, 54, 13],
    [13, 35, 54],
    [13, 35, 14],
]