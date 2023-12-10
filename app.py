import gradio as gr
from detectron2.config import get_cfg
import torch
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer as Visualizer
import tempfile
import shutil

# Function to process video
def process_video(input_video_path):
    # Temporary path for output video
    output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    # Initialize Detectron2 configuration for DensePose
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file("detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml")
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        with torch.no_grad():
            outputs = predictor(frame)['instances']
        
        results = DensePoseResultExtractor()(outputs)
        cmap = cv2.COLORMAP_VIRIDIS
        # Visualizer outputs black for background, but we want the 0 value of
        # the colormap, so we initialize the array with that value
        arr = cv2.applyColorMap(np.zeros((height, width), dtype=np.uint8), cmap)
        out_frame = Visualizer(alpha=1, cmap=cmap).visualize(arr, results)        
        out.write(out_frame)

    # Release resources
    cap.release()
    out.release()

    # Return processed video
    return output_video_path

# Gradio interface
iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Input Video"),
    outputs=gr.Video(label="Output DensePose Video"),
    title="Video 2 DensePose"
)

# Run the app
iface.launch()
