import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import goa_loader.download_data as download_lib
from goa_loader.path import goa_loader_root
import goa_loader.util as util_lib

def load(base_dir=None, download=True, force_download=False, image_size=(200, 200)):
    base_dir = base_dir or goa_loader_root
    base_dir = os.path.abspath(base_dir)
    csv_file = f"{goa_loader_root}/data/annotations/published_images.csv"

    if force_download:
        download_lib.download(base_dir=base_dir)
    if not os.path.exists(csv_file):
        if download:
            download_lib.download(base_dir=base_dir)
        else:
            raise ValueError(
                f"csv_file not found, {csv_file}. "
                "Try running `goa_loader.load(download=True)."
            )

    df = pd.read_csv(csv_file)
    df['local_path'] = df.apply(lambda row: util_lib.thumbnail_to_local(base_dir, row.iiifthumburl), axis=1)
    ds = tf.data.Dataset.from_tensor_slices(df['local_path'])

    resize = tf.keras.layers.Resizing(image_size[0], image_size[1])
    def process_image(path):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)
        return resize(img)

    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds
