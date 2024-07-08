# RL Sandbox

This is my sandbox where I'm trying to understand how to train a Tensorflow
model to control a robot to do things in Mujoco. There are many like it, but
this one is mine.

Right now I'm just trying to get the damn thing to find a red box and drive
towards it. Maybe even avoid some obstacles on the way.

## Installation

```console
$ git clone https://github.com/adammck/rl-sandbox.git
$ pyenv install
$ pip install -e .
```

## Usage

Generate some training data:\
(This puts some examples in `data/img`, and a lot more in `data/tfds`.)

```console
$ mjpython ./sandbox/cmd/gendata/main.py
```

The train the vision model:\
(This writes the output model to pwd.)

```console
$ python ./sandbox/cmd/train/main.py
```

To infer the position of the target (a bright red box) in an image:

```console
$ python ./sandbox/cmd/infer/main.py *.keras data/img/x1y1.png
```
