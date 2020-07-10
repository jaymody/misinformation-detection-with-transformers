import os
import copy
import json
import argparse
from pathlib import Path


def phase2_to_test_data(metadata_file, output_dir, truncate=None):
    with open(metadata_file) as fi:
        metadata = json.load(fi)
    metadata = metadata[:truncate]

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # labelled
    metadata_labelled = {str(claim["id"]): claim for claim in copy.deepcopy(metadata)}
    for claim in metadata_labelled.values():
        del claim["claim"]
        del claim["claimant"]
        del claim["date"]
        del claim["id"]
    with open(os.path.join(output_dir, "metadata-labelled.json"), "w") as fo:
        json.dump(metadata_labelled, fo, indent=2)

    # unlabelled
    metadata_unlabelled = copy.deepcopy(metadata)
    for claim in metadata_unlabelled:
        del claim["label"]
        del claim["related_articles"]
    with open(os.path.join(output_dir, "metadata-unlabelled.json"), "w") as fo:
        json.dump(metadata_unlabelled, fo, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--truncate", type=int, default=None)
    args = parser.parse_args()

    phase2_to_test_data(**args.__dict__)
