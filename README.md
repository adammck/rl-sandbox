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
$ ./rlsb gen --files=10 --tfds=1000
generating 10 files in data/img...
generating tensorflow dataset with 1000 elements in data/tfds...
```

The train the vision model:

```console
$ ./rlsb train --epochs=10
training vision.keras on data/tfds...
Epoch 1/10
200/200 ━━━━━━━━━━━━━━━━━━━━ 177s 868ms/step - accuracy: 0.0497 - loss: 3.9188 - val_accuracy: 0.1700 - val_loss: 3.1028
Epoch 2/10
200/200 ━━━━━━━━━━━━━━━━━━━━ 144s 721ms/step - accuracy: 0.1602 - loss: 3.1381 - val_accuracy: 0.7000 - val_loss: 1.6150
Epoch 3/10
200/200 ━━━━━━━━━━━━━━━━━━━━ 144s 720ms/step - accuracy: 0.5666 - loss: 1.8197 - val_accuracy: 0.8150 - val_loss: 0.7560
(etc)
```

To infer the position of the target (a bright red box) in an image:

```console
$ ./rlsb infer vision.keras data/img/x7y1.png
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 130ms/step
data/img/x7y1.png: x=7, y=1
```
