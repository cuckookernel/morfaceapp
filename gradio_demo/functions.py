from pprint import pformat
from typing import Tuple, TypeAlias

import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw

import backend.retinaface.core as rf_core

from backend import fld
# from backend.gradio.face_lm_detect import NpArray, RETINAFACE, LM_DETECTOR
from backend.utils import BBox, draw_landmarks_ip, draw_triangles, FlatBBox

NpArray: TypeAlias = np.ndarray

# TODO is there a way to avoid these variables being global...?
print('Loading face detector...')
rf_core.cpu = False  # use GPU
RETINAFACE = rf_core.Retinaface()

print('Loading face landmark detector...')
LM_DETECTOR = fld.load_landmarks_model()


def detect_face(input_img: NpArray) -> tuple[str, list[FlatBBox], NpArray, NpArray]:
    face_bboxes0 = RETINAFACE(input_img)
    #  need to convert to regular floats as np.float32 types 'is not json serializable'...
    face_bboxes = [[float(x) for x in bbox] for bbox in face_bboxes0]
    print(f"Face detector result: {face_bboxes}")
    int_bbox = [ int(x) for x in face_bboxes[0]]

    bboxes_text = f"""
    ## Face Detection results
    ```
        bounding_boxes = {pformat([[np.round(float(x),2) for x in bbox] for bbox in face_bboxes0])}
    ```"""

    return (bboxes_text,
            face_bboxes[0],
            draw_bbox(input_img, int_bbox),
            crop_image(input_img, int_bbox))


def draw_bbox(input_img: NpArray, int_bbox: list[int]) -> NpArray:

    pil_image = Image.fromarray(input_img)
    draw = ImageDraw(pil_image)
    # w, h = int_bbox[2] - int_bbox[0], int_bbox[3] - int_bbox[1]

    for i in range(-2, 3):
        draw.rectangle(((int_bbox[0]-i, int_bbox[1]-i), (int_bbox[2]+i, int_bbox[3]+i)),
                       fill=None, outline='#aaffaa')

    return np.array(pil_image)


def crop_image(input_img: NpArray, int_bbox: list[int]) -> NpArray:
    return input_img[int_bbox[1]:int_bbox[3],
                     int_bbox[0]:int_bbox[2]]


def detect_landmarks(full_image: NpArray, face_bbox: list[float], display_mode: str) \
        -> Tuple[NpArray, BBox, NpArray]:
    print(f'detect_landmarks: full_image.shape={full_image.shape}, face_bbox={face_bbox}')
    landmarks, lm_bbox = fld.process_1_face(np.array(face_bbox),
                                            full_image, LM_DETECTOR)

    if display_mode == 'Landmarks':
        img_with_lms = draw_landmarks_ip(full_image, None, landmarks)
    elif display_mode == 'Triangles':
        img_with_lms = draw_triangles(full_image, None, landmarks, fld.TRIANGLES,
                                      color=(100, 100, 255), thickness=1)
    else:
        img_with_lms = full_image

    cropped_img = lm_bbox.with_margin(img_with_lms.shape, 0.15).as_int().crop_img(img_with_lms)

    return landmarks, lm_bbox, cropped_img
