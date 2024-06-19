import os
from unittest.mock import patch
import spaces
import gradio as gr
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports
import torch
import requests
from PIL import Image, ImageDraw
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import io

def workaround_fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

with patch("transformers.dynamic_module_utils.get_imports", workaround_fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True).to("cuda").eval()
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

colormap = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'red',
            'lime', 'indigo', 'violet', 'aqua', 'magenta', 'coral', 'gold', 'tan', 'skyblue']

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

@spaces.GPU
def run_example(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.size[0], image.size[1])
    )
    return parsed_answer

def plot_bbox(image, data):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='indigo', alpha=0.5))
    ax.axis('off')
    return fig_to_pil(fig)

def draw_polygons(image, prediction, fill_mask=False):
    fig, ax = plt.subplots()
    ax.imshow(image)
    scale = 1
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if _polygon.shape[0] < 3:
                print('Invalid polygon:', _polygon)
                continue
            _polygon = (_polygon * scale).reshape(-1).tolist()
            if len(_polygon) % 2 != 0:
                print('Invalid polygon:', _polygon)
                continue
            polygon_points = np.array(_polygon).reshape(-1, 2)
            if fill_mask:
                polygon = patches.Polygon(polygon_points, edgecolor=color, facecolor=fill_color, linewidth=2)
            else:
                polygon = patches.Polygon(polygon_points, edgecolor=color, fill=False, linewidth=2)
            ax.add_patch(polygon)
        plt.text(polygon_points[0, 0], polygon_points[0, 1], label, color='white', fontsize=8, bbox=dict(facecolor=color, alpha=0.5))
    ax.axis('off')
    return fig_to_pil(fig)

def draw_ocr_bboxes(image, prediction):
    fig, ax = plt.subplots()
    ax.imshow(image)
    scale = 1
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        polygon = patches.Polygon(new_box, edgecolor=color, fill=False, linewidth=3)
        ax.add_patch(polygon)
        plt.text(new_box[0], new_box[1], label, color='white', fontsize=8, bbox=dict(facecolor=color, alpha=0.5))
    ax.axis('off')
    return fig_to_pil(fig)


@spaces.GPU(duration=120)
def process_video(input_video_path, task_prompt):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Can't open the video file.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output_vid.mp4", fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        result = run_example(task_prompt, pil_image)

        if task_prompt == "<OD>":
            processed_image = plot_bbox(pil_image, result['<OD>'])
        elif task_prompt == "<DENSE_REGION_CAPTION>":
            processed_image = plot_bbox(pil_image, result['<DENSE_REGION_CAPTION>'])
        else:
            processed_image = pil_image

        processed_frame = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
        out.write(processed_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return "output_vid.mp4"

css = """
#output {
    min-height: 100px;
    overflow: auto;
    border: 1px solid #ccc;
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML("<h1><center>Microsoft Florence-2-large-ft</center></h1>")
    with gr.Tab(label="Image"):
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Input Picture", type="pil")
                task_radio = gr.Radio(
                    ["Caption", "Detailed Caption", "More Detailed Caption", "Caption to Phrase Grounding",
                     "Object Detection", "Dense Region Caption", "Region Proposal", "Referring Expression Segmentation",
                     "Region to Segmentation", "Open Vocabulary Detection", "Region to Category", "Region to Description",
                     "OCR", "OCR with Region"],
                    label="Task", value="Caption"
                )
                text_input = gr.Textbox(label="Text Input (is Optional)", visible=False)
                submit_btn = gr.Button(value="Submit")
            with gr.Column():
                output_text = gr.Textbox(label="Results")
                output_image = gr.Image(label="Image", type="pil")

    with gr.Tab(label="Video"):
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label="Video")
                video_task_radio = gr.Radio(
                    ["Object Detection", "Dense Region Caption"],
                    label="Video Task", value="Object Detection"
                )
                video_submit_btn = gr.Button(value="Process Video")
            with gr.Column():
                output_video = gr.Video(label="Video")

    def update_text_input(task):
        return gr.update(visible=task in ["Caption to Phrase Grounding", "Referring Expression Segmentation",
                                           "Region to Segmentation", "Open Vocabulary Detection", "Region to Category",
                                           "Region to Description"])

    task_radio.change(fn=update_text_input, inputs=task_radio, outputs=text_input)

    def process_image(image, task, text):
        task_mapping = {
            "Caption": ("<CAPTION>", lambda result: (result['<CAPTION>'], image)),
            "Detailed Caption": ("<DETAILED_CAPTION>", lambda result: (result['<DETAILED_CAPTION>'], image)),
            "More Detailed Caption": ("<MORE_DETAILED_CAPTION>", lambda result: (result['<MORE_DETAILED_CAPTION>'], image)),
            "Caption to Phrase Grounding": ("<CAPTION_TO_PHRASE_GROUNDING>", lambda result: (str(result['<CAPTION_TO_PHRASE_GROUNDING>']), plot_bbox(image, result['<CAPTION_TO_PHRASE_GROUNDING>']))),
            "Object Detection": ("<OD>", lambda result: (str(result['<OD>']), plot_bbox(image, result['<OD>']))),
            "Dense Region Caption": ("<DENSE_REGION_CAPTION>", lambda result: (str(result['<DENSE_REGION_CAPTION>']), plot_bbox(image, result['<DENSE_REGION_CAPTION>']))),
            "Region Proposal": ("<REGION_PROPOSAL>", lambda result: (str(result['<REGION_PROPOSAL>']), plot_bbox(image, result['<REGION_PROPOSAL>']))),
            "Referring Expression Segmentation": ("<REFERRING_EXPRESSION_SEGMENTATION>", lambda result: (str(result['<REFERRING_EXPRESSION_SEGMENTATION>']), draw_polygons(image, result['<REFERRING_EXPRESSION_SEGMENTATION>'], fill_mask=True))),
            "Region to Segmentation": ("<REGION_TO_SEGMENTATION>", lambda result: (str(result['<REGION_TO_SEGMENTATION>']), draw_polygons(image, result['<REGION_TO_SEGMENTATION>'], fill_mask=True))),
            "Open Vocabulary Detection": ("<OPEN_VOCABULARY_DETECTION>", lambda result: (str(convert_to_od_format(result['<OPEN_VOCABULARY_DETECTION>'])), plot_bbox(image, convert_to_od_format(result['<OPEN_VOCABULARY_DETECTION>'])))),
            "Region to Category": ("<REGION_TO_CATEGORY>", lambda result: (result['<REGION_TO_CATEGORY>'], image)),
            "Region to Description": ("<REGION_TO_DESCRIPTION>", lambda result: (result['<REGION_TO_DESCRIPTION>'], image)),
            "OCR": ("<OCR>", lambda result: (result['<OCR>'], image)),
            "OCR with Region": ("<OCR_WITH_REGION>", lambda result: (str(result['<OCR_WITH_REGION>']), draw_ocr_bboxes(image, result['<OCR_WITH_REGION>']))),
        }

        if task in task_mapping:
            prompt, process_func = task_mapping[task]
            result = run_example(prompt, image, text)
            return process_func(result)
        else:
            return "", image

    submit_btn.click(fn=process_image, inputs=[input_img, task_radio, text_input], outputs=[output_text, output_image])
    video_submit_btn.click(fn=process_video, inputs=[input_video, video_task_radio], outputs=output_video)

demo.launch()