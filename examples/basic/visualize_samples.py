import matplotlib.pyplot as plt

import goa_loader


def gallery_show(images):
    images = images.numpy()
    images = images.astype(int)
    plt.figure(figsize=(4, 4))
    for i in range(9):
        image = images[i]
        plt.subplot(3, 3, i + 1)
        plt.imshow(image.astype("uint8"))
        plt.axis("off")
    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


dataset = goa_loader.load()
dataset = dataset.batch(9)

for batch in dataset.take(10):
    gallery_show(batch)
