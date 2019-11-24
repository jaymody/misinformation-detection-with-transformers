#### Setup ####
# Python Standard Library
import logging
import argparse

# Third Party Packages
import pandas as pd

# Project Imports
from preprocess import preprocess
from generate_support import generate_support
from predict import basic_config, load_model, predict

# CLI
parser = argparse.ArgumentParser("Predict true, partly true, or false, for fake news data.")
parser.add_argument("--data_path", type=str)
parser.add_argument("--articles_dir", type=str)
parser.add_argument("--predictions_fpath", type=str)
parser.add_argument("--model_dir", type=str)
parser.add_argument("--word2vec_path", type=str)
parser.add_argument("--keep_n", type=int)
parser.add_argument("--nproc", type=int)
parser.add_argument("--ngpu", type=int)

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s: %(asctime)s: %(message)s')
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
logger.handlers = [sh]


#### Main ####
def main(data_path, articles_dir, predictions_fpath, model_dir, word2vec_path, keep_n, nproc, ngpu):
    ## Preproccess
    data, articles = preprocess(data_path, articles_dir, nproc)
    
    ## Create Examples
    samples = generate_support(data, articles, word2vec_path, keep_n, nproc)
    examples = []
    for sample in samples:
        for support in sample["support"]:
            examples.append({
                "text": sample["claim"] + " " + support["text"],
                "id": sample["id"]
            })
    
    ## Load Model
    model_args = basic_config("outputs/", "cache/", nproc=nproc, ngpu=ngpu)
    model = load_model(model_dir, model_args)
    
    ## Predict
    df = pd.DataFrame.from_records(examples)
    predictions = predict(df["text"], df["id"], model)
    
    ## Write Predictions
    with open(predictions_fpath, 'w', encoding="utf-8") as fo:
        for claim_id, prediction in predictions.items():
            fo.write("%d,%d\n" % (claim_id, prediction))
    
    ## End of Main
    logger.info("DONE main()")


#### Main Call ####
if __name__ == "__main__":
    args = parser.parse_args()
    main(**args.__dict__)
