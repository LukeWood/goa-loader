# Gallery of Art Loader

![Demo image](media/demo-image.png)

`goa_loader.load()` loads a dataset of images from the
[National Gallery of Art Open Data Program](https://github.com/NationalGalleryOfArt/opendata)
into a `tf.data.Dataset`.
This dataset may be used for anything; from generative modeling to style transfer.
Check out _Quickstart_ or `examples/` to see how you can get started.

## National Gallery of Art Open Data Program

[The National Gallery of Art Open Data Program has an official Github repo](https://github.com/NationalGalleryOfArt/opendata)

The [National Gallery of Art](https://www.nga.gov) serves the United States by welcoming all people to explore and experience art, creativity, and our shared humanity. In pursuing our mission, we are making certain data about our collection available to scholars, educators, and the general public in CSV format to support research, teaching, and personal enrichment; to promote interdisciplinary research; to encourage fun and creativity; and to help people understand the inspiration behind great works of art. We hope that access to this dataset will fuel knowledge, scholarship, and innovation, inspiring uses that transform the way we discover and understand the world of art.

To the extent permitted by law, the National Gallery of Art waives any copyright or related rights that it might have in this dataset and is releasing this dataset under the [Creative Commons Zero](https://creativecommons.org/publicdomain/zero/1.0/) designation.

The dataset provides data records relating to the 130,000+ artworks in our collection and the artists who created them.  You can download the dataset free of charge without seeking authorization from the National Gallery of Art.  

## Quickstart

Getting started with the `goa_loader` loader is as easy as:

```bash
git clone https://github.com/lukewood/goa-loader
cd goa-loader
python setup.py develop
```

Then you can load the dataset with:

```python
dataset = goa_loader.load()
```

To make sure your installation works, try out:

```
python examples/visualize_samples.py
```

## Citation

TODO
