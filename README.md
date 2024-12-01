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

To generate the protobuf client:

```console
$ pip install ".[dev]"
$ bin/fetch-proto.sh
$ bin/gen-proto.sh
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

04. Launch a simulator and use [collector][] to drive the robot to the target:
  
    ```console
    $ mjpython rlsb capture vision.keras
    probs:[[
    0.01756358 0.00339703 0.04447755 0.00732891 0.01335713 0.00801716 0.01445346 0.00182443
    0.03114431 0.03054641 0.04737312 0.03249083 0.04354941 0.04958116 0.04125877 0.03696582
    0.04325326 0.00793735 0.00630351 0.01282464 0.05445582 0.02147534 0.02284948 0.04686137
    0.01520264 0.00874409 0.00221496 0.01566958 0.00640734 0.00537212 0.01209291 0.00607654
    0.01367156 0.00638251 0.01081797 0.00667950 0.00721032 0.01604502 0.01073315 0.01917819
    0.00251368 0.00944842 0.00201012 0.00213599 0.01380129 0.01155413 0.01865303 0.02611676
    0.00393350 0.00337499 0.00085456 0.00625737 0.00589379 0.00324253 0.00651496 0.00758196
    0.00684720 0.00760710 0.00930837 0.00603096 0.00577906 0.00912645 0.00488668 0.02473890 ]]
    m:0.054455824196338654
    d:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    waiting...
    ```

    It's just dumping state into the console. No red boxes here.
    (meanwhile, elsewhere)

    ```console
    $ pwd
    /Users/adammck/code/src/github.com/adammck/collector
    $ go run main.go
    2024/10/13 22:13:54 Input: input/*.jsonl
    2024/10/13 22:13:54 Listening on :8000
    ```

    This runs the simulator, and at each interval (every second or so), pauses
    it, grabs the pixels from the (simulated) camera, runs the vision model on
    them to (hopefully) find the red box, then writes the state somewhere that
    [collector][] can find it and waits for some human to provide an action in
    response.

    This is all horribly complicated, but I'm trying to support use-cases more
    complex than driving towards red boxes in otherwise empty rooms.

05. Convert the resulting data into tfds format:

    ```console
    $ ./rlsb convert
    ```

    Yeah this is pretty janky, sorry. I haven't figured out how it should work.

06. Use that data to train a control model:

    ```console
    $ python rlsb train-control --epochs 100
    training control model control.keras on data/control...
    Epoch 1/100
    33/33 [==============================] - 1s 8ms/step - loss: 1.6302 - accuracy: 0.2016 - val_loss: 1.6001 - val_accuracy: 0.3125
    (etc)
    ```

07. Run the simulator using the control model:

    ```console
    $ mjpython rlsb run --cam=overview
    ```

[collector]: https://github.com/adammck/collector
