import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_cv import bounding_box
from tqdm.auto import tqdm

import goa_loader.download_data as download_lib
from goa_loader.path import goa_loader_root


def load(base_dir=None, download=True):
    base_dir = data_dir or goa_loader_root
    base_dir = os.path.abspath(base_dir)
    csv_file = f"{goa_loader_root}/data/annotations/published_images.csv"

    if not os.path.exists(csv_file):
        download_lib.download(base_dir=data_dir)
    ds, ds_info = load_scisrs_dataset(
        data_path, csv_path, bounding_box_format=bounding_box_format
    )
    if batch_size is not None:
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size))
    return ds, ds_info
