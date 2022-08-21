## Denoising Diffusion Implicit Models

![Learning process of the DDIM](media/learning.gif)

## Quickstart

Getting started is as easy as:

```
python train.py
```

More advanced usage can be done with the CLI flags:

```bash
python train.py --help

flags:

train.py:
  --artifacts_dir: artifact save dir
  --checkpoint_path: model checkpoint directory
    (default: 'artifacts/checkpoint/diffusion_model')
  --epochs: epochs to train for
    (default: '100')
    (an integer)
  --[no]force_download: Whether or not to force download the dataset.
    (default: 'false')
  --model_dir: directory to save model to
  --percent: percentage of dataset to use
    (default: '100.0')
    (a number)
```
