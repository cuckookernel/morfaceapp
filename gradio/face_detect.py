

import numpy as np
import gradio as gr

from backend.retinaface.core import Retinaface
import backend.retinaface.core as rf_core

rf_core.cpu = False  # use GPU
retinaface = Retinaface()


def detect_face(input_img):
    face_bboxes0 = retinaface(input_img)
    #  need to convert to regular floats as np.float32 types 'is not json serializable'...
    face_bboxes = [[float(x) for x in bbox] for bbox in face_bboxes0]
    print(f"Face detector result: {face_bboxes}")

    return input_img


demo = gr.Interface(detect_face, gr.Image(), outputs="image")
demo.launch()
