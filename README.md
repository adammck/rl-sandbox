# RL Sandbox

This is my sandbox where I'm trying to understand how to train a Tensorflow
model to control a robot to do things in Mujoco. There are many like it, but
this one is mine.

Right now I'm just trying to get the damn thing to find a red box and drive
towards it.

## Installation

```console
$ git clone https://github.com/adammck/rl-sandbox.git
$ pyenv install
$ pip install -e .
```

## Usage

Generate some training data:

```console
$ mjpython ./sandbox/cmd/gendata/main.py
```

The train the model:

```console
$ python ./sandbox/cmd/train/main.py
```
