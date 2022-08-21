import argparse
import os
import urllib.request
import zipfile
from math import floor

import pandas as pd
import tqdm
from joblib import Parallel, delayed

import goa_loader.util as util_lib
from goa_loader.path import get_base_dir

csv_remote_path = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/published_images.csv"


def get_file(remote_url, out, timeout_seconds=10):
    with urllib.request.urlopen(remote_url, timeout=timeout_seconds) as response:
        with open(out, "wb") as out_file:
            data = response.read()  # a `bytes` object
            out_file.write(data)


def download(base_dir=None, percent=100):
    print("Downloading data...")

    base_dir = base_dir or get_base_dir()
    csv_file = f"{base_dir}/annotations/published_images.csv"

    if not os.path.exists(csv_file):
        print(f"CSV not found, downloading from {csv_remote_path}")
        get_file(csv_remote_path, out=csv_file, timeout_seconds=100)
        print("Download successful")

    print(f"Reading annotations from {csv_file}")
    df = pd.read_csv(csv_file)

    samples = floor(df.shape[0] * (percent / 100))
    print(f"Found {df.iiifthumburl.nunique()} images.")
    print(f"Downloading {samples}/{df.shape[0]} images")

    df = df.head(samples)
    print("Downloading images...")

    def download(thumb):
        out = util_lib.thumbnail_to_local(base_dir, thumb)
        if os.path.exists(out):
            return

        try:
            get_file(thumb, out=out)
        except RuntimeError as e:
            print(e)
            print(f"failed to get {thumb}")

    results = Parallel(n_jobs=16)(
        delayed(download)(thumb) for thumb in tqdm.tqdm(df.iiifthumburl.unique())
    )
    print("Done downloading images")


def main():
    parser = argparse.ArgumentParser(description="download gallery of art data")
    parser.add_argument("--base_dir", "-b", default="./")

    args = parser.parse_args()
    if not args.base_dir:
        parser.print_help()
        quit()

    download(base_dir=args.base_dir)


if __name__ == "__main__":
    main()
