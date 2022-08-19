import argparse
import zipfile

import requests
import os
from goa_loader.path import goa_loader_root
import wget
import pandas as pd
import tqdm
from joblib import Parallel, delayed

csv_remote_path = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/published_images.csv"

def download(base_dir=None):
    print("Downloading data...")

    base_dir = base_dir or goa_loader_root
    csv_file = f"{goa_loader_root}/data/annotations/published_images.csv"

    if not os.path.exists(csv_file):
        print(f"CSV not found, downloading from {csv_remote_path}")
        wget.download(csv_remote_path, out=csv_file)
        print("Download successful")

    print(f"Reading annotations from {csv_file}")
    df = pd.read_csv(csv_file)

    image_path=f"{goa_loader_root}/data/images"
    print(f"Found {df.iiifthumburl.nunique()} images.")

    print("Downloading images...")

    def download(thumb):
        ending = "_".join(thumb.split("/")[-5:])
        out = f"{image_path}/{ending}"
        wget.download(thumb, out=out)

    results = Parallel(n_jobs=16)(delayed(download)(thumb) for thumb in tqdm.tqdm(df.iiifthumburl.unique()))
    print("Done downloading images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download gallery of art data")
    parser.add_argument("--base_dir", "-b", default="./")

    args = parser.parse_args()
    if not args.base_dir:
        parser.print_help()
        quit()

    download(base_dir=args.base_dir)
