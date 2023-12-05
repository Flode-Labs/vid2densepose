# Vid2DensePose
<a target="_blank" href="https://colab.research.google.com/drive/1x77dESn7EGPCqjKdQ1sJJhNu0Cf-5Gpt?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Overview

The Vid2DensePose is a powerful tool designed for applying the DensePose model to videos, generating detailed "Part Index" visualizations for each frame. This tool is exceptionally useful for enhancing animations, particularly when used in conjunction with MagicAnimate for temporally consistent human image animation.

## Key Features


- **Enhanced Output**: Produces video files showcasing DensePosedata in a vivid, color-coded format.
- **MagicAnimate Integration**: Seamlessly compatible with MagicAnimate to foster advanced human animation projects.

## Prerequisites

To utilize this tool, ensure the installation of:
- Python 3.8 or later
- PyTorch (preferably with CUDA for GPU support)
- Detectron2

## Installation Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/flode/vid2densepose.git
    cd vid2densepose
    ```

2. Install necessary Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Clone the Detectron repository:
    ```bash
        git clone https://github.com/facebookresearch/detectron2.git
    ```

## Usage Guide

1. Modify the `main.py` script to set your desired input (`INPUT_VIDEO_PATH`) and output (`OUTPUT_VIDEO_PATH`) video paths.

2. Run the script:
    ```bash
    python main.py 
    ```

The script processes the input video and generates an output with the densePose format.

## Integration with MagicAnimate

For integration with MagicAnimate:

1. Create the densepose video using the steps outlined above.
2. Use this output as an input to MagicAnimate for generating temporally consistent animations.


## Acknowledgments

Special thanks to:
- Facebook AI Research (FAIR) for the development of DensePose.
- The contributors of the Detectron2 project.

## Support

For any inquiries or support, please file an issue in our GitHub repository's issue tracker.

