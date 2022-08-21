import os

from os.path import expanduser

home = expanduser("~")
goa_loader_root = f"{home}/data/goa-loader"
goa_loader_root = os.path.abspath(goa_loader_root)


def ensure_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_base_dir():
    ensure_exists(goa_loader_root)
    ensure_exists(f"{goa_loader_root}/annotations")
    ensure_exists(f"{goa_loader_root}/images")
    return goa_loader_root
