import os
from unittest.mock import patch
import spaces
import gradio as gr
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports
import torch
from PIL import Image, ImageDraw
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import io
import uuid

def workaround_fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with patch("transformers.dynamic_module_utils.get_imports", workaround_fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True).to(device).eval()
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

colormap = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'red',
            'lime', 'indigo', 'violet', 'aqua', 'magenta', 'coral', 'gold', 'tan', 'skyblue']

def run_example(task_prompt, image, text_input=None):
    prompt = task_prompt if text_input is None else task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=1024, early_stopping=False, do_sample=False, num_beams=3)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.size[0], image.size[1]))

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)

def plot_bbox_img(image, data):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    
    if 'bboxes' in data and 'labels' in data:
        bboxes, labels = data['bboxes'], data['labels']
    elif 'bboxes' in data and 'bboxes_labels' in data:
        bboxes, labels = data['bboxes'], data['bboxes_labels']
    else:
        return fig_to_pil(fig)

    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='indigo', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=10, bbox=dict(facecolor='indigo', alpha=0.8))
    
    ax.axis('off')
    return fig_to_pil(fig)

def draw_poly_img(image, prediction, fill_mask=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    for polygons, label in zip(prediction.get('polygons', []), prediction.get('labels', [])):
        color = random.choice(colormap)
        for polygon in polygons:
            if isinstance(polygon[0], (int, float)):
                polygon = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
            poly = patches.Polygon(polygon, edgecolor=color, facecolor=color if fill_mask else 'none', alpha=0.5 if fill_mask else 1, linewidth=2)
            ax.add_patch(poly)
        if polygon:
            plt.text(polygon[0][0], polygon[0][1], label, color='white', fontsize=10, bbox=dict(facecolor=color, alpha=0.8))
    ax.axis('off')
    return fig_to_pil(fig)

def draw_ocr_bboxes(image, prediction):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        box_array = np.array(box).reshape(-1, 2)  # respect format
        polygon = patches.Polygon(box_array, edgecolor=color, fill=False, linewidth=2)
        ax.add_patch(polygon)
        plt.text(box_array[0, 0], box_array[0, 1], label, color='white', fontsize=10, bbox=dict(facecolor=color, alpha=0.8))
    ax.axis('off')
    return fig_to_pil(fig)

def plot_bbox(image, data):
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), label, fill="white")
    return np.array(img_draw)

@spaces.GPU(duration=120)
def process_video(input_video_path, task_prompt):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return None, []

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    result_file_name = f"{uuid.uuid4()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_file_name, fourcc, fps, (frame_width, frame_height))

    processed_frames = 0
    frame_results = []
    color_map = {}  #consistency for chromakey possibility

    def get_color(label):
        if label not in color_map:
            color_map[label] = random.choice(colormap)
        return color_map[label]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        try:
            result = run_example(task_prompt, pil_image)

            if task_prompt == "<OD>":
                processed_image = plot_bbox(pil_image, result['<OD>'])
                frame_results.append((processed_frames + 1, result['<OD>']))
            elif task_prompt == "<DENSE_REGION_CAPTION>":
                processed_image = pil_image.copy()
                draw = ImageDraw.Draw(processed_image)
                for i, label in enumerate(result['<DENSE_REGION_CAPTION>'].get('labels', [])):
                    draw.text((10, 10 + i*20), label, fill="white")
                processed_image = np.array(processed_image)
                frame_results.append((processed_frames + 1, result['<DENSE_REGION_CAPTION>']))
            elif task_prompt in ["<REFERRING_EXPRESSION_SEGMENTATION>", "<REGION_TO_SEGMENTATION>"]:
                if isinstance(result[task_prompt], dict) and 'polygons' in result[task_prompt]:
                    processed_image = draw_vid_polygons(pil_image, result[task_prompt], get_color)
                else:
                    processed_image = np.array(pil_image)
                frame_results.append((processed_frames + 1, result[task_prompt]))
            else:
                processed_image = np.array(pil_image)

            out.write(cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
            processed_frames += 1

        except Exception as e:
            print(f"Error processing frame {processed_frames + 1}: {str(e)}")
            processed_image = np.array(pil_image)
            out.write(cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
            processed_frames += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if processed_frames == 0:
        return None, frame_results

    return result_file_name, frame_results

def draw_vid_polygons(image, prediction, get_color):
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    for polygons, label in zip(prediction.get('polygons', []), prediction.get('labels', [])):
        color = get_color(label)
        for polygon in polygons:
            if isinstance(polygon[0], (int, float)): 
                polygon = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
            draw.polygon(polygon, outline=color, fill=color)
        if polygon:
            draw.text(polygon[0], label, fill="white")
    return np.array(img_draw)

def process_image(image, task, text):
    task_mapping = {
        "Caption": ("<CAPTION>", lambda result: (result['<CAPTION>'], image)),
        "Detailed Caption": ("<DETAILED_CAPTION>", lambda result: (result['<DETAILED_CAPTION>'], image)),
        "More Detailed Caption": ("<MORE_DETAILED_CAPTION>", lambda result: (result['<MORE_DETAILED_CAPTION>'], image)),
        "Caption to Phrase Grounding": ("<CAPTION_TO_PHRASE_GROUNDING>", lambda result: (str(result['<CAPTION_TO_PHRASE_GROUNDING>']), plot_bbox_img(image, result['<CAPTION_TO_PHRASE_GROUNDING>']))),
        "Object Detection": ("<OD>", lambda result: (str(result['<OD>']), plot_bbox_img(image, result['<OD>']))),
        "Dense Region Caption": ("<DENSE_REGION_CAPTION>", lambda result: (str(result['<DENSE_REGION_CAPTION>']), plot_bbox_img(image, result['<DENSE_REGION_CAPTION>']))),
        "Region Proposal": ("<REGION_PROPOSAL>", lambda result: (str(result['<REGION_PROPOSAL>']), plot_bbox_img(image, result['<REGION_PROPOSAL>']))),
        "Referring Expression Segmentation": ("<REFERRING_EXPRESSION_SEGMENTATION>", lambda result: (str(result['<REFERRING_EXPRESSION_SEGMENTATION>']), draw_poly_img(image, result['<REFERRING_EXPRESSION_SEGMENTATION>'], fill_mask=True))),
        "Region to Segmentation": ("<REGION_TO_SEGMENTATION>", lambda result: (str(result['<REGION_TO_SEGMENTATION>']), draw_poly_img(image, result['<REGION_TO_SEGMENTATION>'], fill_mask=True))),
        "Open Vocabulary Detection": ("<OPEN_VOCABULARY_DETECTION>", lambda result: (str(result['<OPEN_VOCABULARY_DETECTION>']), plot_bbox_img(image, result['<OPEN_VOCABULARY_DETECTION>']))),
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

def map_task_to_prompt(task):
    task_mapping = {
        "Object Detection": "<OD>",
        "Dense Region Caption": "<DENSE_REGION_CAPTION>",
        "Referring Expression Segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
        "Region to Segmentation": "<REGION_TO_SEGMENTATION>"
    }
    return task_mapping.get(task, "")

def process_video_p(input_video, task, text_input):
    prompt = map_task_to_prompt(task)
    if task == "Referring Expression Segmentation" and text_input:
        prompt += text_input
    result, frame_results = process_video(input_video, prompt)
    if result is None:
        return None, "Error: Video processing failed. Check logs above for info.", str(frame_results)
    return result, result, str(frame_results)

with gr.Blocks() as demo:
    gr.HTML("<h1><center>Microsoft Florence-2-large-ft</center></h1>")
    
    with gr.Tab(label="Image"):
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Input Picture", type="pil")
                task_dropdown = gr.Dropdown(
                    choices=["Caption", "Detailed Caption", "More Detailed Caption", "Caption to Phrase Grounding",
                             "Object Detection", "Dense Region Caption", "Region Proposal", "Referring Expression Segmentation",
                             "Region to Segmentation", "Open Vocabulary Detection", "Region to Category", "Region to Description",
                             "OCR", "OCR with Region"],
                    label="Task", value="Caption"
                )
                text_input = gr.Textbox(label="Text Input (is Optional)", visible=False)
                gr.Examples(
                    examples=[
                        [
                            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true",
                            "Detailed Caption",
                            "",
                        ],
                        [
                            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true",
                            "Object Detection",
                            "",
                        ],
                        [
                            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true",
                            "Caption to Phrase Grounding",
                            "A green car parked in front of a yellow building."
                        ],
                        [
                            "http://ecx.images-amazon.com/images/I/51UUzBDAMsL.jpg?download=true",
                            "OCR",
                            ""
                        ]
                    ],
                    inputs=[input_img, task_dropdown, text_input],
                )
                submit_btn = gr.Button(value="Submit")
            with gr.Column():
                output_text = gr.Textbox(label="Results")
                output_image = gr.Image(label="Image", type="pil")

    with gr.Tab(label="Video"):
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label="Video")
                video_task_dropdown = gr.Dropdown(
                    choices=["Object Detection", "Dense Region Caption", "Referring Expression Segmentation", "Region to Segmentation"],
                    label="Video Task", value="Object Detection"
                )
                video_text_input = gr.Textbox(label="Text Input (for Referring Expression Segmentation)", visible=False)
                video_submit_btn = gr.Button(value="Process Video")
            with gr.Column():
                output_video = gr.Video(label="Processed Video")
                frame_results_output = gr.Textbox(label="Frame Results")

    def update_text_input(task):
        return gr.update(visible=task in ["Caption to Phrase Grounding", "Referring Expression Segmentation",
                                           "Region to Segmentation", "Open Vocabulary Detection", "Region to Category",
                                           "Region to Description"])

    task_dropdown.change(fn=update_text_input, inputs=task_dropdown, outputs=text_input)

    def update_video_text_input(task):
        return gr.update(visible=task == "Referring Expression Segmentation")

    video_task_dropdown.change(fn=update_video_text_input, inputs=video_task_dropdown, outputs=video_text_input)

    submit_btn.click(fn=process_image, inputs=[input_img, task_dropdown, text_input], outputs=[output_text, output_image])
    video_submit_btn.click(fn=process_video_p, inputs=[input_video, video_task_dropdown, video_text_input], outputs=[output_video, output_video, frame_results_output])

demo.launch()