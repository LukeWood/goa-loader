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

splits = {
    "train": "train.csv",
    "test": "test.csv",
    "val": "val.csv",
    "validation": "val.csv",
}


def parse_annotation_file(annotation_filename):

    # First entry in a line is the label, other entries are bbox coordinates
    labels = []
    bboxes = []
    with open(annotation_filename) as f:
        for line in f:
            line = line.split()
            # array.append([float(x) for x in line.split()])
            labels.append(int(line[0]))
            bboxes.append([float(x) for x in line[1:]])

    return (np.array(bboxes), np.array(labels))


def load_scisrs_dataset(base_path, csv_path, bounding_box_format):
    # Load the given csv that contains image and annotation filenames
    df = pd.read_csv(csv_path)

    # Add complete paths to image and annotation files
    df["image_path"] = base_path + "/" + df["image_filename"].astype(str)
    df["annotation_path"] = base_path + "/" + df["annotation_filename"].astype(str)

    image_paths = df["image_path"]
    annotation_paths = df["annotation_path"]

    def dataset_generator():
        for image_path, annotation_path in zip(image_paths, annotation_paths):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            # img /= 255.0 # It loooks like this might be done in a preprocessing stage

            annotation = parse_annotation_file(annotation_path)
            yolo_bboxes = annotation[0]
            labels = annotation[1]

            bboxes = np.zeros(yolo_bboxes.shape)
            bboxes[:, 0] = (
                yolo_bboxes[:, 1] - yolo_bboxes[:, 3] / 2
            )  # y1 = y_center - h/2
            bboxes[:, 1] = (
                yolo_bboxes[:, 0] - yolo_bboxes[:, 2] / 2
            )  # x1 = x_center - w/2
            bboxes[:, 2] = (
                yolo_bboxes[:, 1] + yolo_bboxes[:, 3] / 2
            )  # y2 = y_center + h/2
            bboxes[:, 3] = (
                yolo_bboxes[:, 0] + yolo_bboxes[:, 2] / 2
            )  # x2 = x_center + w/2

            yield (img, bboxes, labels)

    def format_data(image, bounding_boxes, class_ids):
        class_ids = tf.expand_dims(tf.cast(class_ids, bounding_boxes.dtype), axis=-1)
        bounding_boxes = tf.concat([bounding_boxes, class_ids], axis=-1)
        return {
            "images": image,
            "bounding_boxes": bounding_box.convert_format(
                bounding_boxes, source="rel_yxyx", target=bounding_box_format, images=image
            ),
        }

    return tf.data.Dataset.from_generator(
        dataset_generator,
        output_signature=(
            tf.TensorSpec(shape=(500, 500, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ),
    ).map(format_data, num_parallel_calls=tf.data.AUTOTUNE), {"num_samples": len(df)}


def load(
    bounding_box_format,
    batch_size=None,
    split="train",
    data_dir=None,
    download=True,
    verbosity=1,
    with_info=True,
):
    base_dir = data_dir or goa_loader_root
    base_dir = os.path.abspath(base_dir)
    data_path = f"{base_dir}/data/version-1"

    if not split in splits:
        raise ValueError(f"Received invalid split, want one of {splits.keys()}")
    if not os.path.exists(data_path):
        if download:
            print("[Warning] Data not found locally, downloading dataset...")
            download_lib.download(base_dir=data_dir)
        else:
            raise ValueError(
                f"{data_path} does not exist, please download the dataset."
            )

    csv_path = f"{base_dir}/data/metadata/{splits[split]}"

    ds, ds_info = load_scisrs_dataset(
        data_path, csv_path, bounding_box_format=bounding_box_format
    )
    if batch_size is not None:
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size))
    return ds, ds_info
