"""
Title: Train a RetinaNet to Detect ElectroMagnetic Signals
Author: [lukewood](https://lukewood.xyz), Kevin Anderson, Peter Gerstoft
Date created: 2022/08/16
Last modified: 2022/08/16
Description:
"""

"""
## Overview
"""

import sys

import keras_cv
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import flags
from keras_cv import bounding_box
from tensorflow import keras
from tensorflow.keras import callbacks as callbacks_lib
from tensorflow.keras import optimizers

import goa_loader
import wandb

flags.DEFINE_integer("batch_size", 8, "Training and eval batch size.")
flags.DEFINE_integer("epochs", 1, "Number of training epochs.")
flags.DEFINE_string("wandb_entity", "scisrs", "wandb entity to use.")
flags.DEFINE_string("experiment_name", None, "wandb run name to use.")
FLAGS = flags.FLAGS

FLAGS(sys.argv)

if FLAGS.wandb_entity:
    wandb.init(
        project="scisrs",
        entity=FLAGS.wandb_entity,
        name=FLAGS.experiment_name,
    )

"""
## Data loading

TODO(lukewood): write this
"""

dataset, dataset_info = goa_loader.load(
    split="train", bounding_box_format="xywh", batch_size=9
)


def visualize_dataset(dataset, bounding_box_format):
    color = tf.constant(((255.0, 0, 0),))
    plt.figure(figsize=(7, 7))
    iterator = iter(dataset)
    for i in range(9):
        example = next(iterator)
        images, boxes = example["images"], example["bounding_boxes"]
        boxes = keras_cv.bounding_box.convert_format(
            boxes, source=bounding_box_format, target="rel_yxyx", images=images
        )
        boxes = boxes.to_tensor(default_value=-1)
        plotted_images = tf.image.draw_bounding_boxes(images, boxes[..., :4], color)
        plt.subplot(9 // 3, 9 // 3, i + 1)
        plt.imshow(plotted_images[0].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()


visualize_dataset(dataset, bounding_box_format="xywh")

"""
Looks like everything is structured as expected.  Now we can move on to constructing our
data augmentation pipeline.
"""

"""
## Data augmentation
"""

# train_ds is batched as a (images, bounding_boxes) tuple
# bounding_boxes are ragged
train_ds, train_dataset_info = goa_loader.load(
    bounding_box_format="xywh", split="train", batch_size=FLAGS.batch_size
)
val_ds, val_dataset_info = goa_loader.load(
    bounding_box_format="xywh", split="val", batch_size=FLAGS.batch_size
)

augmentation_layers = [
    # keras_cv.layers.RandomShear(x_factor=0.1, bounding_box_format='xywh'),
    # TODO(lukewood): add color jitter and others
]


def augment(sample):
    for layer in augmentation_layers:
        sample = layer(sample)
    return sample


train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
visualize_dataset(train_ds, bounding_box_format="xywh")

"""
Great!  We now have a bounding box friendly augmentation pipeline.

Next, let's unpackage our inputs from the preprocessing dictionary, and prepare to feed
the inputs into our model.
"""


def unpackage_dict(inputs):
    return inputs["images"], inputs["bounding_boxes"]


train_ds = train_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

"""
Our data pipeline is now complete.  We can now move on to model creation and training.
"""

"""
## Model creation

We'll use the KerasCV API to construct a RetinaNet model.  In this tutorial we use
a pretrained ResNet50 backbone using weights.  In order to perform fine-tuning, we
freeze the backbone before training.  When `include_rescaling=True` is set, inputs to
the model are expected to be in the range `[0, 255]`.
"""

model = keras_cv.models.RetinaNet(
    classes=20,
    bounding_box_format="xywh",
    backbone="resnet50",
    backbone_weights="imagenet",
    include_rescaling=True,
)

"""
That is all it takes to construct a KerasCV RetinaNet.  The RetinaNet accepts tuples of
dense image Tensors and ragged bounding box Tensors to `fit()` and `train_on_batch()`
This matches what we have constructed in our input pipeline above.

The RetinaNet `call()` method outputs two values: training targets and inference targets.
In this guide, we are primarily concerned with the inference targets.  Internally, the
training targets are used by `keras_cv.losses.ObjectDetectionLoss()` to train the
network.
"""

"""
## Optimizer

For training, we use a SGD optimizer with a piece-wise learning rate schedule
consisting of a warm up followed by a ramp up, then a ramp.
Below, we construct this using a `keras.optimizers.schedules.PiecewiseConstantDecay`
schedule.
"""

optimizer = optimizers.SGD(learning_rate=0.1, momentum=0.9, global_clipnorm=10.0)

"""
## COCO metrics monitoring

"""

metrics = [
    keras_cv.metrics.COCOMeanAveragePrecision(
        class_ids=range(20),
        bounding_box_format="xywh",
        name="Mean Average Precision",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=range(20),
        bounding_box_format="xywh",
        max_detections=100,
        name="Recall",
    ),
]

"""
## Training our model

All that is left to do is train our model.  KerasCV object detection models follow the
standard Keras workflow, leveraging `compile()` and `fit()`.

Let's compile our model:
"""

loss = keras_cv.losses.ObjectDetectionLoss(
    classes=20,
    classification_loss=keras_cv.losses.FocalLoss(from_logits=True, reduction="none"),
    box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
    reduction="auto",
)

model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=metrics,
)

"""
All that is left to do is construct some callbacks:
"""

callbacks = [
    callbacks_lib.TensorBoard(log_dir="logs"),
    callbacks_lib.EarlyStopping(patience=50),
    callbacks_lib.ReduceLROnPlateau(patience=20),
]
if FLAGS.wandb_entity:
    callbacks += [
        wandb.keras.WandbCallback(save_model=False),
    ]

"""
And run `model.fit()`!
"""

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FLAGS.epochs,
    callbacks=callbacks,
)

"""
## Results and conclusions
"""
