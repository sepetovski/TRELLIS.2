# opencv-python 5.x wheels ship without OpenEXR, so assets/hdri/*.exr fail to load.
pip install pillow imageio imageio-ffmpeg tqdm easydict "opencv-python-headless>=4.10,<5" pandas open3d objaverse huggingface_hub[cli] open_clip_torch
