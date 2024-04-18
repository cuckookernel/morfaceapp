import cv2
import numpy as np
import gradio as gr

from backend import fld
from backend.face_comb import ProcessedImage, combine
from backend.gradio.functions import LM_DETECTOR, NpArray, RETINAFACE
from backend.utils import BBox, Landmarks, draw_landmarks_ip, draw_triangles


def detect_landmarks_v2(input_image: NpArray) -> tuple[ProcessedImage, NpArray, NpArray]:
    print(f'detect_landmarks_v2: input_image.shape={input_image.shape},')

    face_bboxes0 = RETINAFACE(input_image)
    #  need to convert to regular floats as np.float32 types 'is not json serializable'...
    face_bboxes = [[float(x) for x in bbox] for bbox in face_bboxes0]
    face_bbox = face_bboxes[0]

    landmarks, lm_bbox = fld.process_1_face(np.array(face_bbox), input_image, LM_DETECTOR)

    img_with_lms = draw_triangles(input_image, None, landmarks, fld.TRIANGLES,
                                  color=(100, 100, 255), thickness=1)

    cropped_img = lm_bbox.with_margin(input_image.shape, 0.15).as_int().crop_img(input_image)
    cropped_img_lms = lm_bbox.with_margin(img_with_lms.shape, 0.15).as_int().crop_img(img_with_lms)

    return ProcessedImage(img=input_image, bbox=lm_bbox,
                          landmarks=landmarks), cropped_img, cropped_img_lms


def save_movie(out_fpath: str,
               proc_img1: ProcessedImage, proc_img2: ProcessedImage,
               progress=gr.Progress()) -> str:

    sequence = []

    lambda_seq = np.concatenate([np.arange(0, 1.0, 0.04), np.arange(1.0, 0., -0.04)])
    for lambda_ in progress.tqdm(lambda_seq):
        img = combine(proc_img1, proc_img2, lambda_)
        sequence.append(img)

    sequence[0].save(out_fpath, save_all=True, append_images=sequence[1:])

    return f"Done!"


with gr.Blocks() as demo:
    proc_img1 = gr.State()
    proc_img2 = gr.State()

    with gr.Row():
        img1 = gr.Image(label='Input Image 1')
        img2 = gr.Image(label='Input Image 2')

    detect_btn = gr.Button('Detect LandMarks')
    with gr.Row():
        img1_with_lms = gr.Image()
        img2_with_lms = gr.Image()

    lambda_ = gr.Slider(minimum=0, maximum=1.0, step=0.025, value=0.5)
    with gr.Row():
        cropped_img1 = gr.Image()
        combined_img = gr.Image(type='pil')
        cropped_img2 = gr.Image()

    detect_btn.click(detect_landmarks_v2,
                     inputs=img1,
                     outputs=[proc_img1, cropped_img1, img1_with_lms])

    detect_btn.click(detect_landmarks_v2,
                     inputs=img2,
                     outputs=[proc_img2, cropped_img2, img2_with_lms])

    lambda_.change(combine,
                   inputs=[proc_img1, proc_img2, lambda_],
                   outputs=combined_img)

    with gr.Row():
        gr.Label('Save animation to file:')
        out_fpath_input = gr.Text('./movie.webp')
        save_webp_btn = gr.Button('Save animation')
        save_notif = gr.Markdown()

        save_webp_btn.click(save_movie,
                            inputs=[out_fpath_input, proc_img1, proc_img2],
                            outputs=save_notif)


demo.launch()
