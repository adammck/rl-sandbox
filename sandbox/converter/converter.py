#!/usr/bin/env python

import tensorflow as tf
import json

def create_dataset(jsonl_path: str):

    # read each line as a separate json document.
    with open(jsonl_path) as f:
        data = [json.loads(line) for line in f]

    # TODO: check the shape of these.
    inputs = [item['input'] for item in data]
    labels = [item['output'] for item in data]

    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    return dataset

def convert(input: str, output: str):
    dataset = create_dataset(input)
    dataset.save(output)
