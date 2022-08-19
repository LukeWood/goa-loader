# Gallery of Art Loader

`goa_loader.load()` loads a dataset of images from the 
[National Gallery of Art Open Data Program](https://github.com/NationalGalleryOfArt/opendata)
into a `tf.data.Dataset`.
This dataset may be used for anything; from generative modeling to style transfer.
Check out _Quickstart_ or `examples/` to see how you can get started.

## Quickstart

Getting started with the `goa_loader` loader is as easy as:

```bash
git clone https://github.com/lukewood/goa-loader
cd goa-loader
python setup.py develop
```

```python
dataset = goa_loader.load()
```

