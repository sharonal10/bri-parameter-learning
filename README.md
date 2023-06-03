# parameter-learning

## Steps to run:
1) Use Python3.11 and create a virtual environment using `python311 -m virtualenv penv` to ensure there are no conflicts with previously installed packages.
2) After activating the virtualenv, run `pip install -r requirements.txt` to install required packages. 
3) To use `<image-file-path>` and `<task-name>` as input, run `python311 sample_run.py <image-file-path> <task-name>`. It will print the predicted coordinate for that task on that image + 1000 normally distributed samples with variance=1. Note that you might see additional output such as `xFormers not available`.

## Key functions 
`pipeline.run`: returns the predicted coordinate, given the filepath of the test image and a task name.

`pipeline.generate_samples`: returns normally distributed samples given a coordinate. Default variance is 1, default number of samples returned is 1000.
