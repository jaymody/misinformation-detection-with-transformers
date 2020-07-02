import os
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trail_data_raw_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    with open(os.path.join(args.trail_data_raw_dir, "2_trial_metadata.json")) as fi:
        trial_metadata_unlabelled = json.load(fi)
    with open(os.path.join(args.trail_data_raw_dir, "2_trial_labels.json")) as fi:
        trial_labels = json.load(fi)
    with open(os.path.join(args.trail_data_raw_dir, "jay-mody.json")) as fi:
        trial_predictions = json.load(fi)

    trial_metadata = [
        {
            **claim,
            "label": trial_labels[str(claim["id"])]["label"],
            "related_articles": trial_labels[str(claim["id"])]["related_articles"],
        }
        for claim in trial_metadata_unlabelled
    ]

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as fo:
        json.dump(trial_metadata, fo, indent=2)
    with open(os.path.join(args.output_dir, "predictions.json"), "w") as fo:
        json.dump(trial_predictions, fo, indent=2)
