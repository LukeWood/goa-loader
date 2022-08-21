import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from tensorflow import keras
from keras import layers
import goa_loader
from model import DiffusionModel
from absl import flags
import sys
import visualization as visualiation_lib

flags.DEFINE_string("artifacts_dir", None, "artifact save dir")
flags.DEFINE_string("checkpoints/diffusion_model", None, "model checkpoint directory")
flags.DEFINE_integer("percent", 100, "percentage of dataset to use")
FLAGS = flags.FLAGS
FLAGS(sys.argv)

artifacts_dir = FLAGS.artifacts_dir or "artifacts"
# data
dataset_repetitions = 5
num_epochs = 5  # train for at least 50 epochs for good results
image_size = 64
# KID = Kernel Inception Distance, see related section
kid_diffusion_steps = 5
plot_diffusion_steps = 20

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
widths = [32, 64, 96, 128]
block_depth = 2

# optimization
batch_size = 64
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4


def preprocess_image(image):
    return tf.clip_by_value(image / 255.0, 0.0, 1.0)


# load dataset
train_dataset = goa_loader.load(image_size=(64, 64), percent=FLAGS.percent)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(10 * batch_size)
train_dataset = train_dataset.repeat(dataset_repetitions)
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

model = DiffusionModel(image_size, widths, block_depth)
model.compile(
    optimizer=tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    loss=keras.losses.mean_absolute_error,
)

# save the best model based on the validation KID metric
checkpoint_path = FLAGS.checkpoint_path
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="val_kid",
    mode="min",
    save_best_only=True,
)

# calculate mean and variance of training dataset for normalization
model.normalizer.adapt(train_dataset)

# run training and plot generated images periodically
model.fit(
    train_dataset,
    epochs=50,
    # validation_data=val_dataset,
    callbacks=[
        visualiation_lib.SaveVisualOfSameNoiseEveryEpoch(
            model=model, save_path=f"{artifacts_dir}/same-noise"
        ),
        visualiation_lib.SaveRandomNoiseImages(
            model=model, save_path=f"{artifacts_dir}/random"
        ),
        checkpoint_callback,
    ],
)
