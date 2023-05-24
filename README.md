# parameter-learning

Steps to run:
1) Use Python3.11 and create a virtual environment using `python311 -m virtualenv penv` to ensure there are no conflicts with previously installed packages.
2) After activating the virtualenv, run `pip install -r requirements.txt` to install required packages. 
3) You will also need to download the ViT-B SAM model from https://github.com/facebookresearch/segment-anything and copy it into this repo, it is too large to put on Github.
4) To use `<image-file-path>` and `<task-name>` as input, run `python311 pipeline.py <image-file-path> <task-name>`. It will print the predicted coordinate for that task on that image. Note that you might see additional output such as `xFormers not available`.