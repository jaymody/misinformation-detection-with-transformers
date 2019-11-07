#### Setup ####
# Python Standard Library
import os
import copy
import glob
import heapq
import json
import logging
import argparse
import collections
import unicodedata
import multiprocessing

# Third Party Imports
import nltk
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from simpletransformers.model import TransformerModel

# CLI
parser = argparse.ArgumentParser("Predict true, partly true, or false, for fake news data.")
parser.add_argument("--data_json", type=str)
parser.add_argument("--articles_dir", type=str)
parser.add_argument("--predictions_fpath", type=str)
parser.add_argument("--model_dir", type=str)
parser.add_argument("--nproc", type=int)
parser.add_argument("--ngpu", type=int)

# Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s: %(asctime)s: %(message)s')
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
logger.handlers = [sh]
logger.info('*** Logging Configurated ***')



#### Preprocess ####
def _clean_text(text):
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output).strip()

def _is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def _is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False

def _split_sentences(text):
    return [sentence.strip() for sentence in nltk.tokenize.sent_tokenize(text)]

def _split_and_clean(x):
    return x[0], _split_sentences(_clean_text(x[1]))

def preprocess(data_json, articles_dir, nproc=1):
    # Load data
    logger.info("... loading train data ...")
    with open(data_json, 'r') as fi:
        raw_data = json.load(fi)
        
    logger.info("... loading articles data ...")
    raw_articles_data = {}
    for fpath in tqdm(glob.glob(os.path.join(articles_dir, "*.txt"))):
        with open(fpath, 'r') as fi:
            raw_articles_data[os.path.basename(fpath).split(".")[0]] = fi.read()
    
    # Clean claim text in training data
    logger.info("... cleaning train data ...")
    data = copy.deepcopy(raw_data)
    for example in tqdm(data):
        example["claim"] = _clean_text(example["claim"])
    
    # Split articles into sentences and clean text
    logger.info("... cleaning articles data ...")
    
    global articles_data
    articles_data = {}
    pool = multiprocessing.Pool(nproc)

    for k,v in tqdm(pool.imap_unordered(_split_and_clean, raw_articles_data.items()), total=len(raw_articles_data)):
        articles_data[k] = v
    
    return data



#### Create Examples ####
def create_examples_from_claim(sample, keep_n=5):
    # Get sentences from related_articles
    sentences = [sentence for ref in sample["related_articles"] for sentence in articles_data[str(ref)]]
    
    # Append and pad claim sentence
    sentences.append(sample["claim"])

    # Get tf_idf vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences)
    
    # Calcualte and sort cosine similarities
    examples = []
    for sentence, vector in zip(sentences, vectors):
        examples.append({
          "id": sample["id"],
          "text": sample["claim"] + " " + sentence,
          "cosine_similarity": float(cosine_similarity(vectors[-1], vector))
        })
    examples.pop()
    examples = heapq.nlargest(keep_n, examples, key=lambda x: x["cosine_similarity"])
    
    return examples

def generate_examples(data, nproc=1):    
    # Generate examples
    logger.info("... generating examples ...")
    all_examples = []
    pool = multiprocessing.Pool(nproc)
    
    for examples in tqdm(pool.imap_unordered(create_examples_from_claim, data), total=len(data)):
        all_examples.extend(examples)

    return all_examples



#### Predict Classes ####
def load_model(model_dir, nproc=1, ngpu=0):
    model_args = {
        "output_dir": "outputs/",
        "cache_dir": "cache/",

        "fp16": False,
        "fp16_opt_level": "O1",
        "max_seq_length": 256,
        "train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "eval_batch_size": 8,
        "num_train_epochs": 1,
        "weight_decay": 0,
        "learning_rate": 2e-5,
        "adam_epsilon": 1e-8,
        "warmup_ratio": 0.06,
        "warmup_steps": 500,
        "max_grad_norm": 1.0,

        "logging_steps": 100,
        "save_steps": 2000,

        "overwrite_output_dir": False,
        "reprocess_input_data": False,

        "process_count": nproc,
        "n_gpu": ngpu,
    }

    return TransformerModel('xlnet', model_dir, num_labels=3, args=model_args)
  
def predict_proba(examples_text, claim_ids, model):
    model_outputs = model.predict(examples_text)
    
    probs = collections.defaultdict(list)
    for claim_id, prob in zip(claim_ids, model_outputs[:][1]):
        probs[claim_id].append(prob)
        
    averaged_probs = {k: np.mean(v, axis=0) for k,v in probs.items()}
    return averaged_probs

def predict(examples_text, claim_ids, model):
    probs = predict_proba(examples_text, claim_ids, model)
    return {k: np.argmax(v) for k,v in probs.items()}



#### Main ####
def main(data_json, articles_dir, predictions_fpath, model_dir, nproc, ngpu):
    ## Preproc
    # WARNING: articles_data becomes global namespace watch out!
    data = preprocess(data_json, articles_dir, nproc)
    
    ## Gen examples
    examples = generate_examples(data, nproc)
    
    ## Load Model
    model = load_model(model_dir, nproc, ngpu)
    
    ## Predict
    df = pd.DataFrame.from_records(examples)
    predictions = predict(df["text"], df["id"], model)
    
    ## Write predictions to output path
    with open(predictions_fpath, 'w') as fo:
        for claim_id, prediction in predictions.items():
            fo.write("%d,%d\n" % (claim_id,prediction))
    
    ## Log main function done executing
    logger.info("DONE main()")

    
if __name__ == "__main__":
    args = parser.parse_args()
    main(**args.__dict__)
