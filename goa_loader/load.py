import json
import os
import random
from math import floor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

import goa_loader.download_data as download_lib
import goa_loader.util as util_lib
from goa_loader.path import get_base_dir


def load(
    base_dir=None,
    download=True,
    force_download=False,
    image_size=(200, 200),
    percent=100,
):
    base_dir = base_dir or get_base_dir()
    base_dir = os.path.abspath(base_dir)
    csv_file = f"{base_dir}/annotations/published_images.csv"

    if force_download:
        download_lib.download(base_dir=base_dir, percent=percent)
    if not os.path.exists(csv_file):
        if download:
            download_lib.download(base_dir=base_dir, percent=percent)
        else:
            raise ValueError(
                f"csv_file not found, {csv_file}. "
                "Try running `goa_loader.load(download=True)."
            )

    df = pd.read_csv(csv_file)
    df["local_path"] = df.apply(
        lambda row: util_lib.thumbnail_to_local(base_dir, row.iiifthumburl), axis=1
    )

    samples = floor(df.shape[0] * (percent / 100))
    print(f"Loading {samples}/{df.shape[0]} images")
    df = df.head(samples)
    ds = tf.data.Dataset.from_tensor_slices(df["local_path"])

    resize = tf.keras.layers.Resizing(image_size[0], image_size[1])

    def process_image(path):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)
        return resize(img)

    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds
