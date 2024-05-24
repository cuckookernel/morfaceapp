
import numpy as np
from PIL import Image, ImageDraw2
from pprint import pformat

from PIL.ImageDraw import ImageDraw

from typing import TypeAlias
import gradio as gr

from backend.retinaface.core import Retinaface
import backend.retinaface.core as rf_core

rf_core.cpu = False  # use GPU
retinaface = Retinaface()

NpArray: TypeAlias = np.ndarray


def detect_face(input_img: NpArray):
    face_bboxes0 = retinaface(input_img)
    #  need to convert to regular floats as np.float32 types 'is not json serializable'...
    face_bboxes = [[float(x) for x in bbox] for bbox in face_bboxes0]
    print(f"Face detector result: {face_bboxes}")
    int_bbox = [ int(x) for x in face_bboxes[0]]

    bboxes_text = f"""```
        bounding_boxes = {pformat([[np.round(float(x),2) for x in bbox] for bbox in face_bboxes0])}
    ```"""

    return bboxes_text, draw_bbox(input_img, int_bbox), crop_image(input_img, int_bbox)


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


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image = gr.Image(label='Input Image')
        with gr.Column():
            with gr.Row():
                bboxes_text = gr.Markdown()
            with gr.Row():
                image_with_bbox = gr.Image(scale=0.5, label='Painted BBox')
            with gr.Row():
                just_face = gr.Image(scale=0.5, label='Cropped Face')

    with gr.Row():
        detect_btn = gr.Button("Detect Face")
        detect_btn.click(fn=detect_face,
                         inputs=image,
                         outputs=[
                            bboxes_text,
                            image_with_bbox,
                            just_face] )

demo.launch()
