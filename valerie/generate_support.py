#### Setup ####
# Python Standard Library
import gc
import heapq
import logging
import multiprocessing

# Third Party Packages
import nltk
import gensim
import numpy as np
from tqdm import tqdm
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s: %(asctime)s: %(message)s')
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
logger.handlers = [sh]


#### Create Support for Single Claim ####
def generate_support_from_claim(sample, keep_n=8, min_threshold=0.40, min_examples=4):
    # Get sentences from related_articles
    corpus = [(ref,sentence) for ref in sample["related_articles"] for sentence in articles_data[str(ref)]]
    references, sentences = map(list, zip(*corpus))
    
    # Append and pad claim sentence
    sentences.append(sample["claim"])
    references.append(None)

    # Get tf_idf vectors
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectors = tfidf_vectorizer.fit_transform(sentences)

    # Get sentence word2vec vectors
    def word2vec_sentence(sentence):
        words = nltk.tokenize.word_tokenize(sentence)
        vectors = []
        for word in words:
            try:
                vectors.append(word2vec_model[word])
            except:
                continue
        return np.nan if not vectors else np.mean(vectors, axis=0)
    word2vec_vectors = [word2vec_sentence(sentence) for sentence in sentences]
    
    # Calculate and sort cosine similarities for embeddings
    support = []
    for ref, sentence, tfidf_vector, word2vec_vector in zip(references, sentences, tfidf_vectors, word2vec_vectors):
        support.append({
            "source_article": ref,
            "text": sentence,
            "scores": {
                "tfidf": float(cosine_similarity(tfidf_vectors[-1], tfidf_vector)),
                "word2vec": 0.0 if np.isnan(np.min(word2vec_vector)) else float(1 - spatial.distance.cosine(word2vec_vectors[-1], word2vec_vector))
            }
        })
    # Calculate overall similarity (mean of cosine similarities)
    for s in support:
        s["score"] = float(np.mean(list(s["scores"].values()), axis=0))
    support.pop() # remove the claim sentence itself from support

    # Sort or get nlargest
    if keep_n:
        support = heapq.nlargest(keep_n, support, key=lambda x: x["score"])
        support = sorted(support, key=lambda x: x["score"], reverse=True)
        support = [s for i,s in enumerate(support) if s["score"] >= min_threshold or i < min_examples]
    else:
        support = sorted(support, key=lambda x: x["score"], reverse=True)
    
    # Update and return sample
    sample["support"] = support
    return sample

#### Load word2vec Model ####
def load_word2vec(word2vec_path):
    return gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True) 

#### Create Support for All Data ####
def generate_support(data, _articles_data, word2vec_path, keep_n=5, nproc=1): 
    # Make articles_data global
    global articles_data
    articles_data = _articles_data

    # Load and globalize word2vec model
    logger.info("... loading word2vec model ...")
    global word2vec_model
    word2vec_model = load_word2vec(word2vec_path)

    # Multiprocessing
    pool = multiprocessing.Pool(nproc)

    # Generate support for each claim
    logger.info("... generating support ...")
    samples = []
    for sample in tqdm(pool.imap_unordered(generate_support_from_claim, data), total=len(data)):
        samples.append(sample)
    
    return samples


### Main ###
def main(data_path, articles_dir, output_fpath, word2vec_path, keep_n, nproc, ngpu):
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
    
    ## Predict
    logger.info("... saving examples to dataframe ...")
    df = pd.DataFrame.from_records(examples)
    df.to_csv(output_fpath)


### Main Call ###
if __name__ == "__main__":
    # Main Imports
    import argparse
    import pandas as pd
    from preprocess import preprocess

    # CLI
    parser = argparse.ArgumentParser("Predict true, partly true, or false, for fake news data.")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--articles_dir", type=str)
    parser.add_argument("--output_fpath", type=str)
    parser.add_argument("--word2vec_path", type=str)
    parser.add_argument("--keep_n", type=int)
    parser.add_argument("--nproc", type=int)
    parser.add_argument("--ngpu", type=int)

    args = parser.parse_args()
    main(**args.__dict__)
