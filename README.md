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

01. Generate some training data for the vision model:

    ```console
    $ ./rlsb gen --files=10 --tfds=1000
    generating 10 files in data/img...
    generating tensorflow dataset with 1000 elements in data/tfds...
    ```

    This puts some examples in `data/img`, and a lot more in `data/tfds`.

02. Train the vision model:

    ```console
    $ ./rlsb train-vision --epochs=10
    training vision.keras on data/tfds...
    Epoch 1/10
    200/200 ━━━━━━━━━━━━━━━━━━━━ 177s 868ms/step - accuracy: 0.0497 - loss: 3.9188 - val_accuracy: 0.1700 - val_loss: 3.1028
    Epoch 2/10
    200/200 ━━━━━━━━━━━━━━━━━━━━ 144s 721ms/step - accuracy: 0.1602 - loss: 3.1381 - val_accuracy: 0.7000 - val_loss: 1.6150
    Epoch 3/10
    200/200 ━━━━━━━━━━━━━━━━━━━━ 144s 720ms/step - accuracy: 0.5666 - loss: 1.8197 - val_accuracy: 0.8150 - val_loss: 0.7560
    (etc)
    ```

03. Infer the position of the target (a bright red box) in an image:

    ```console
    $ ./rlsb infer vision.keras data/img/x7y1.png
    1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 130ms/step
    data/img/x7y1.png: x=7, y=1
    ```

04. Launch a simulator and capture some data to train the control model:
  
    ```console
    $ mjpython rlsb capture vision.keras
    x=3, y=1 -> act=0
    x=3, y=1 -> act=0
    (etc)
    ```

    This runs the simulator, and allows the user to control the robot with the
    arrow keys. Every sampling interval, it captures the view from the camera,
    runs it through the vision model to produce an approximate (x, y) position
    of the target, and captures that along with the most recent keycode.
