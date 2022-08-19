import goa_loader
import matplotlib.pyplot as plt

def gallery_show(images):
    images = images.numpy()
    images = images.astype(int)
    for i in range(9):
        image = images[i]
        plt.subplot(3, 3, i + 1)
        plt.imshow(image.astype("uint8"))
        plt.axis("off")
    plt.show()

dataset = goa_loader.load()
dataset = dataset.batch(9)

gallery_show(next(iter(dataset)))
