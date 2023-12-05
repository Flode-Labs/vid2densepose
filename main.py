from detectron2.config import get_cfg
import torch
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer as Visualizer

# Initialize Detectron2 configuration for DensePose
cfg = get_cfg()
add_densepose_config(cfg)
cfg.merge_from_file("detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml")
cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
predictor = DefaultPredictor(cfg)

# Video paths
INPUT_VIDEO_PATH = "./input_video.mp4"
OUTPUT_VIDEO_PATH = "./output_video.mp4"

# Open the input video
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    with torch.no_grad():
        outputs = predictor(frame)['instances']
    
    results = DensePoseResultExtractor()(outputs)
    out_frame = Visualizer(alpha=1).visualize(np.zeros(frame.shape, dtype=np.uint8), results)
    
    out.write(out_frame)

# Release resources
cap.release()
out.release()
