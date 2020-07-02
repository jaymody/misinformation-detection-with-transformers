import os
import json
import argparse
from pathlib import Path


def phase2_to_test_data(metadata_file, output_dir, truncate):
    with open(metadata_file) as fi:
        metadata = json.load(fi)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    metadata = metadata[:truncate]
    with open(os.path.join(output_dir, "metadata-labelled.json"), "w") as fo:
        json.dump(metadata, fo, indent=2)

    for claim in metadata:
        del claim["label"]
        del claim["related_articles"]

    with open(os.path.join(output_dir, "metadata-unlabelled.json"), "w") as fo:
        json.dump(metadata, fo, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--truncate", type=int, default=None)
    args = parser.parse_args()

    phase2_to_test_data(**args.__dict__)
