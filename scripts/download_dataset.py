#!/usr/bin/env python3
"""Download verified training/validation episodes from the pinned HF revision."""
import argparse, os, sys
from pathlib import Path
# ros_env.sh intentionally makes inference offline. Retrieval is explicitly online.
os.environ["HF_HUB_OFFLINE"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iss_capture.public_dataset import DEFAULT_DIRECTORY, download_dataset

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--episode", action="append", help="Download only this episode; repeatable")
    parser.add_argument("--split", choices=("train", "validation"))
    args = parser.parse_args()
    print(download_dataset(args.destination, args.episode, args.split))

if __name__ == "__main__":
    main()
