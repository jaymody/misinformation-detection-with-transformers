import json
import argparse

from valerie.scoring import validate_predictions_phase2, compute_detailed_score_phase2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_file", type=str)
    parser.add_argument("--predictions_file", type=str)
    parser.add_argument("--score_file", type=str, default=None)
    parser.add_argument("--report_file", type=str, default=None)
    parser.add_argument("--print", type=bool, default=True)
    args = parser.parse_args()

    with open(args.metadata_file) as fi:
        labels = json.load(fi)
    with open(args.predictions_file) as fi:
        predictions = json.load(fi)

    if not validate_predictions_phase2(predictions)[1]:
        raise ValueError("validate_predictions_phase2 failed")
    report, output, official_output = compute_detailed_score_phase2(labels, predictions)

    if args.print:
        print("score:")
        print(json.dumps(output, indent=2))
        print()
        print("report:")
        print(report)

    if args.report_file:
        with open(args.report_file, "w") as fo:
            fo.write(report)
    if args.score_file:
        with open(args.score_file, "w") as fo:
            json.dump(output, fo, indent=2)
