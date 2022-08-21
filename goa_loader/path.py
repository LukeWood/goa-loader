import os

from os.path import expanduser

home = expanduser("~")
goa_loader_root = f'{home}/data/goa-loader'
goa_loader_root = os.path.abspath(goa_loader_root)

def get_base_dir():
    if not os.path.exists(goa_loader_root):
        os.makedirs(goa_loader_root)
    return goa_loader_root
