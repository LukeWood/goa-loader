import argparse
import zipfile

import os
from goa_loader.path import goa_loader_root
import goa_loader.util as util_lib

import urllib.request
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

    print(f"Found {df.iiifthumburl.nunique()} images.")

    print("Downloading images...")

    def download(thumb):
        out = util_lib.thumbnail_to_local(base_dir, thumb)
        if os.path.exists(out):
            return
        try:
            with urllib.request.urlopen(url, timeout=10) as response, open(file_name, 'wb') as out_file:
                data = response.read() # a `bytes` object
                out_file.write(data)
        except:
            print(f"failed to get {thumb}")

    results = Parallel(n_jobs=16)(delayed(download)(thumb) for thumb in tqdm.tqdm(df.iiifthumburl.unique()))
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
