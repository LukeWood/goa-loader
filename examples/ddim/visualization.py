import os

import matplotlib.pyplot as plt
from tensorflow import keras


def ensure_exists(path):
    if not os.path.exists(path):
        os.path.makedirs(path)


def visualize_and_save_images(images, path, rows=3, cols=6):
    fig = plt.figure(figsize=(cols*2, rows*2))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.axis("off")

    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            plt.subplot(rows, cols, index + 1)
            plt.imshow(images[index])
            plt.axis("off")
            plt.margins(x=0, y=0)

    plt.savefig(fname=path, pad_inches=0, bbox_inches='tight', transparent=True)
    plt.close()


class SaveVisualOfSameNoiseEveryEpoch(keras.callbacks.Callback):
    def __init__(self, model, save_path, rows=3, cols=6, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.save_path = save_path
        self.rows = rows
        self.cols = cols
        self.initial_noise = model.produce_initial_noise(rows * cols)
        ensure_exists(save_path)

    def on_epoch_end(self, epoch, logs=None):
        images = self.model.generate_from_noise(self.initial_noise)
        epoch = str(epoch).zfill(3)
        visualize_and_save_images(images, path=f"{self.save_path}/{epoch}.png")


class SaveRandomNoiseImages(keras.callbacks.Callback):
    def __init__(self, model, save_path, rows=3, cols=6, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.save_path = save_path
        self.rows = rows
        self.cols = cols
        ensure_exists(save_path)

    def on_epoch_end(self, epoch, logs=None):
        images = self.model.generate(self.rows * self.cols)
        epoch = str(epoch).zfill(3)
        visualize_and_save_images(
            images, path=f"{self.save_path}/{epoch}.png", rows=self.rows, cols=self.cols
        )
