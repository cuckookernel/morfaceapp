
import gradio as gr

from gradio_demo.functions import detect_face, detect_landmarks

with gr.Blocks() as demo:
    face_bbox_var = gr.State()
    landmarks_var = gr.State()
    lm_bbox_var = gr.State()

    # Section 1: Face detection
    with gr.Row():
        with gr.Column(scale=2):
            image = gr.Image(label='Input Image')
        with gr.Column(scale=2):
            detect_btn = gr.Button("Detect Face")
            with gr.Row():
                bboxes_text = gr.Markdown()
            with gr.Row():
                image_with_bbox = gr.Image(scale=0.5, label='Painted BBox')
            with gr.Row():
                just_face = gr.Image(scale=0.5, label='Cropped Face')

    # Section 2: Landmark Detection
    with gr.Row():
        with gr.Column():
            copy_input_btn = gr.Button('Copy Face from Step 1')
            fld_input = gr.Image()
        with gr.Column():
            run_fld_btn = gr.Button('Detect Landmarks')
            display_mode = gr.Radio(label='Display Mode', choices=['Landmarks', 'Triangles'])
            fld_output = gr.Image()

    detect_btn.click(fn=detect_face,
                     inputs=image,
                     outputs=[bboxes_text,
                              face_bbox_var,
                              image_with_bbox,
                              just_face])

    copy_input_btn.click(fn=lambda x: x,
                         inputs=just_face,
                         outputs=fld_input)

    run_fld_btn.click(
        fn=detect_landmarks,
        inputs=[image, face_bbox_var, display_mode],
        outputs=[landmarks_var, lm_bbox_var, fld_output]
    )

demo.launch()
