import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_unlabelled_file",
        type=str,
        default="data/phase2-validation/raw/val_metadata_p2.json",
    )
    parser.add_argument(
        "--metadata_labelled_file",
        type=str,
        default="data/phase2-validation/raw/2_labels.json",
    )
    parser.add_argument(
        "--metadata_output_file",
        type=str,
        default="data/phase2-validation/raw/metadata.json",
    )
    args = parser.parse_args()

    with open(args.metadata_unlabelled_file) as fi:
        metadata_unlabelled = json.load(fi)
    with open(args.metadata_labelled_file) as fi:
        metadata_labelled = json.load(fi)

    for claim in metadata_unlabelled:
        claim["label"] = metadata_labelled[str(claim["id"])]["label"]
        claim["related_articles"] = metadata_labelled[str(claim["id"])][
            "related_articles"
        ]

    with open(args.metadata_output_file, "w") as fo:
        json.dump(metadata_unlabelled, fo, indent=2)
