#!/usr/bin/env python

import os
import argparse
from sandbox.generators import random as gen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=int, default=0, help="number of files to generate")
    parser.add_argument("--tfds", type=int, default=0, help="number of elements in tfds")
    parser.add_argument("--files-path", type=str, default="data/img", help="path to files")
    parser.add_argument("--tfds-path", type=str, default="data/tfds", help="path to tfds")
    args = parser.parse_args()

    if args.files > 0:
        print(f"generating {args.files} files in {args.files_path}...")
        gen.generate_files(args.files, args.files_path)
    
    if args.tfds > 0:
        print(f"generating tensorflow dataset with {args.tfds} elements in {args.tfds_path}...")
        gen.generate_dataset(args.tfds, args.tfds_path)

if __name__ == "__main__":
    main()
