# Vid2DensePose
<a target="_blank" href="https://colab.research.google.com/drive/1x77dESn7EGPCqjKdQ1sJJhNu0Cf-5Gpt?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

![](https://github.com/Flode-Labs/vid2densepose/blob/main/sample_videos/side_by_side.gif)

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
    git clone https://github.com/Flode-Labs/vid2densepose.git
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

Run the script:
    
```bash
python main.py -i sample_videos/input_video.mp4 -o sample_videos/output_video.mp4
```

The script processes the input video and generates an output with the densePose format.

####  Gradio version
You can also use the Gradio to run the script with an interface. To do so, run the following command:
```bash
python app.py
```

## Integration with MagicAnimate

For integration with MagicAnimate:

1. Create the densepose video using the steps outlined above.
2. Use this output as an input to MagicAnimate for generating temporally consistent animations.


## Acknowledgments

Special thanks to:
- Facebook AI Research (FAIR) for the development of DensePose.
- The contributors of the Detectron2 project.
- [Gonzalo Vidal](https://www.tiktok.com/@_gonzavidal) for the sample videos.
- [Sylvain Filoni](https://twitter.com/fffiloni) for the deployment of the Gradio Space in [Hugging Face](https://huggingface.co/spaces/fffiloni/video2densepose).

## Support

For any inquiries or support, please file an issue in our GitHub repository's issue tracker.

